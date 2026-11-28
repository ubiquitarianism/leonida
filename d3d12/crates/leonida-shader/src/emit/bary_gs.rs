//! The geometry shader that stands in for D3D12 barycentrics on a host
//! without them (`D3D12-BARYCENTRICS-FALLBACK`; Pascal and older). Such a
//! host refuses a pixel shader that reads `SV_Barycentrics` or calls
//! `attributeAtVertex`, so the translator
//! ([`super::set_host_barycentrics`]) has the pixel shader read ordinary
//! inputs instead, and this stage, behind the translated vertex shader,
//! writes them:
//!
//! - [`super::BARY_INPUT`] (three floats, interpolated as the pixel shader
//!   declares): (1, 0, 0), (0, 1, 0) and (0, 0, 1) at vertices 0, 1 and 2,
//!   which interpolate to the weights `SV_Barycentrics` holds.
//! - [`super::VERTEX_INPUT`] (one constant channel each): vertex 1's or
//!   2's value of the vertex shader output in the register and column its
//!   semantic index names ([`super::vertex_source`]). Vertex 0's is the
//!   constant input itself: the triangle goes out in the order it came, so
//!   the first vertex, whose value a constant input takes, is vertex 0.
//!
//! Every other output is the vertex shader's, register for register, except
//! one in a register the stand-ins take (the pixel shader does not read
//! it: its own inputs come first). A pixel shader that declares a
//! primitive id gets the triangle's, which the host leaves to the geometry
//! stage when there is one.

use super::bitcode::{Attr, Cast, Term, V};
use super::container::SigElement;
use super::shader::{Geometry, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The scalars a geometry shader may emit per invocation (vertices times
/// output components).
const MAX_SCALARS: u32 = 1024;

fn signature(dxil: &[u8], fourcc: FourCC) -> Result<Vec<SigElement>, ShaderError> {
    let c = Container::parse(dxil)?;
    match c.part(fourcc) {
        Some(part) => super::signature(part.data),
        None => Ok(Vec::new()),
    }
}

fn stand_in(e: &SigElement) -> bool {
    e.name == super::BARY_INPUT || e.name == super::VERTEX_INPUT
}

/// Whether the translated pixel shader `ps` ([`super::HostShader::dxil`])
/// reads inputs that only [`bary_gs`] writes.
pub fn needs_bary_gs(ps: &[u8]) -> bool {
    signature(ps, FourCC::ISG1).is_ok_and(|inputs| inputs.iter().any(stand_in))
}

/// The geometry shader (a DXIL container) for triangles drawn with the
/// translated vertex shader `vs` and pixel shader `ps`.
pub fn bary_gs(vs: &[u8], ps: &[u8]) -> Result<Vec<u8>, ShaderError> {
    build(signature(vs, FourCC::OSG1)?, &signature(ps, FourCC::ISG1)?)
}

/// What an output of the stage carries.
enum Source {
    /// The vertex shader's output `id`, as it is.
    Output(usize),
    /// One of the weights: 1 at vertex `k`'s column, 0 elsewhere.
    Weights,
    /// Vertex `k`'s value of input `id`'s column `col`, or 0 when the vertex
    /// shader writes nothing there.
    Vertex(Option<(usize, u8)>, u32),
    /// The primitive id.
    Primitive,
}

fn build(mut vs_outputs: Vec<SigElement>, ps_inputs: &[SigElement]) -> Result<Vec<u8>, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    if vs_outputs.iter().any(|e| e.row.is_none()) {
        return Err(unsupported("a barycentrics stage behind an output without a register"));
    }
    for e in &mut vs_outputs {
        // Interpolation as `emit_program` declares a vertex shader's
        // outputs; a geometry shader's inputs carry the same modes.
        e.interp = match (e.kind, e.comp) {
            (3, _) => 4,
            (0 | 6 | 7, 9) => 2,
            (0, _) => 1,
            _ => 0,
        };
        e.usage = (1 << e.cols) - 1;
    }
    let ours: Vec<&SigElement> = ps_inputs.iter().filter(|e| stand_in(e)).collect();
    if ours.iter().any(|e| e.row.is_none()) {
        return Err(unsupported("a barycentrics stand-in without a register"));
    }
    let taken = |row: Option<u32>| ours.iter().any(|e| e.row == row);

    // The pixel shader placed its stand-ins after its own inputs, not
    // knowing the vertex shader's registers: an output there the pixel
    // shader does not read. A system value (the position, clip distances)
    // moves to a free register past them, where the pixel shader does not
    // look for it; anything else is left out.
    let mut outputs = Vec::new();
    let mut sources = Vec::new();
    let mut moved = Vec::new();
    for (id, e) in vs_outputs.iter().enumerate() {
        if taken(e.row) {
            if e.kind != 0 {
                moved.push(id);
            }
            continue;
        }
        outputs.push(e.clone());
        sources.push(Source::Output(id));
    }
    let used = outputs.iter().chain(ours.iter().copied()).filter_map(|e| e.row).max();
    let mut free = used.map_or(0, |r| r + 1);
    // (elements that shared a row stay together: old row, new row)
    let mut rows: Vec<(Option<u32>, u32)> = Vec::new();
    for id in moved {
        let mut e = vs_outputs[id].clone();
        let row = match rows.iter().find(|(old, _)| *old == e.row) {
            Some(&(_, new)) => new,
            None => {
                rows.push((e.row, free));
                free += 1;
                free - 1
            }
        };
        e.row = Some(row);
        outputs.push(e);
        sources.push(Source::Output(id));
    }
    if free > 32 {
        return Err(unsupported("a barycentrics stage with more than 32 output registers"));
    }
    for e in &ours {
        let source = if e.name == super::BARY_INPUT {
            Source::Weights
        } else {
            let (k, row, col) = super::vertex_source(e.index);
            if !(1..=2).contains(&k) {
                return Err(unsupported("a vertex stand-in of a vertex other than 1 or 2"));
            }
            let input = vs_outputs.iter().enumerate().find_map(|(id, o)| {
                (o.row == Some(row) && col >= o.col && col < o.col + o.cols).then(|| (id, col - o.col))
            });
            Source::Vertex(input, u32::from(k))
        };
        // (a signature has no interpolation mode: as `emit_program`
        // declares the vertex shader's float and integer outputs)
        let interp = if e.name == super::VERTEX_INPUT { 1 } else { 2 };
        outputs.push(SigElement {
            usage: (1 << e.cols) - 1,
            interp,
            ..(*e).clone()
        });
        sources.push(source);
    }
    // The primitive id the pixel shader declares, as one more output.
    let wanted = ps_inputs
        .iter()
        .find(|e| e.kind == 10 && e.row.is_some())
        .filter(|_| !vs_outputs.iter().any(|o| o.kind == 10));
    if let Some(e) = wanted {
        // Alone in its columns, and in a row of constant elements.
        let clash = outputs.iter().any(|o| {
            let overlap = o.col < e.col + e.cols && e.col < o.col + o.cols;
            o.row == e.row && (o.interp != 1 || overlap)
        });
        if clash {
            return Err(unsupported(
                "a barycentrics stage whose primitive id shares a register with another output",
            ));
        }
        outputs.push(SigElement {
            interp: 1,
            usage: (1 << e.cols) - 1,
            ..e.clone()
        });
        sources.push(Source::Primitive);
    }
    let scalars: u32 = outputs.iter().map(|e| u32::from(e.cols)).sum();
    if scalars * 3 > MAX_SCALARS {
        return Err(unsupported(
            "a barycentrics stage with more output components than a geometry shader emits",
        ));
    }

    let mut s = Shader::new(false);
    s.geometry = Some(Geometry { max_vertices: 3 });
    s.inputs = vs_outputs;
    s.outputs = outputs;
    let b = s.entry;
    let primitive = sources
        .iter()
        .any(|x| matches!(x, Source::Primitive))
        .then(|| s.op(b, "primitiveID.i32", Attr::ReadNone, s.t.i32, 108, &[]));
    let stream = s.m.i8(0);
    for vertex in 0..3u32 {
        for (out, source) in sources.iter().enumerate() {
            let float = s.outputs[out].comp == 9;
            let cols = s.outputs[out].cols;
            let values: Vec<V> = match *source {
                Source::Output(id) => (0..cols).map(|col| load(&mut s, id, col, vertex)).collect(),
                Source::Weights => (0..u32::from(cols))
                    .map(|col| s.m.f32((if col == vertex { 1f32 } else { 0f32 }).to_bits()))
                    .collect(),
                Source::Vertex(input, k) => {
                    let v = match input {
                        Some((id, col)) => {
                            let v = load(&mut s, id, col, k);
                            let from_float = s.inputs[id].comp == 9;
                            match (from_float, float) {
                                (true, false) => s.m.cast(b, Cast::Bitcast, v, s.t.i32),
                                (false, true) => s.m.cast(b, Cast::Bitcast, v, s.t.f32),
                                _ => v,
                            }
                        }
                        None if float => s.m.f32(0),
                        None => s.m.i32(0),
                    };
                    vec![v]
                }
                Source::Primitive => vec![primitive.expect("the primitive id is read")],
            };
            let name = if float { "storeOutput.f32" } else { "storeOutput.i32" };
            for (col, value) in values.into_iter().enumerate() {
                let args = [s.m.i32(out as u32), s.m.i32(0), s.m.i8(col as u8), value];
                s.op(b, name, Attr::NoUnwind, s.t.void, 5, &args);
            }
        }
        s.op(b, "emitStream", Attr::NoUnwind, s.t.void, 97, &[stream]);
    }
    s.op(b, "cutStream", Attr::NoUnwind, s.t.void, 98, &[stream]);
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

/// Column `col` of input `id` at vertex `vertex` of the triangle.
fn load(s: &mut Shader, id: usize, col: u8, vertex: u32) -> V {
    let (name, ty) = if s.inputs[id].comp == 9 {
        ("loadInput.f32", s.t.f32)
    } else {
        ("loadInput.i32", s.t.i32)
    };
    let args = [
        s.m.i32(id as u32),
        s.m.i32(0),
        s.m.i8(col),
        s.m.i32(vertex),
    ];
    let entry = s.entry;
    s.op(entry, name, Attr::ReadNone, ty, 4, &args)
}
