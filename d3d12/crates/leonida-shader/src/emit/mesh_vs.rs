//! The vertex shader that draws a mesh program run as compute on a host
//! without mesh shaders (`D3D12-MESH-COMPUTE-FALLBACK`). The compute
//! variant ([`super::set_host_mesh_shaders`]) writes each group's vertices,
//! primitive indices and per-primitive outputs to [`super::SLOT_MESH_OUT`]
//! in its [`MeshOut`] layout; the draw that follows is a triangle list of
//! `3 * max_primitives` vertices and one instance a group, with this shader
//! in front of the title's pixel shader, reading the same buffer as the root
//! SRV [`super::SLOT_MESH_IN`]. (An indirect dispatch draws with the
//! header's arguments, which the groups wrote.)
//!
//! Vertex `id` of instance `g` is corner `id % 3` of primitive `id / 3` of
//! group `g`. A group past [`MeshOut::groups_cap`] had no slot of its own:
//! it draws nothing. A primitive the program culled, or
//! never wrote, has indices 0, 0, 0: a triangle of no area, as on the mesh
//! path. Every corner is its own vertex here, so a per-primitive output is
//! a vertex output with constant interpolation, the same at all three.
//!
//! The outputs are the mesh program's vertex outputs, register for
//! register, and its per-primitive outputs in the registers the pixel
//! shader reads them from (by semantic and index): a mesh pipeline's
//! per-primitive outputs have their own signature, which a vertex shader
//! has not.

use super::bitcode::{Attr, Bin, Term, V};
use super::container::SigElement;
use super::shader::{MeshOut, Shader};
use crate::ShaderError;
use crate::dxbc::{Container, FourCC};

/// The pass-through vertex shader (a DXIL container) for a mesh program
/// translated as compute with `layout`, drawn with the translated pixel
/// shader `ps` (its [`super::HostShader::dxil`]) if the pipeline has one.
pub fn mesh_vs(layout: &MeshOut, ps: Option<&[u8]>) -> Result<Vec<u8>, ShaderError> {
    let vertex = super::signature(&layout.osg1)?;
    let primitive = super::signature(&layout.psg1)?;
    let wanted = match ps {
        Some(ps) => {
            let c = Container::parse(ps)?;
            match c.part(FourCC::ISG1) {
                Some(part) => super::signature(part.data)?,
                None => Vec::new(),
            }
        }
        None => Vec::new(),
    };
    build(layout, vertex, &primitive, &wanted)
}

/// Where an output's value comes from.
enum Source {
    /// Byte `at` of its vertex (`false`) or of its primitive past the
    /// indices (`true`).
    Out { primitive: bool, at: u32 },
    /// [`super::BARY_INPUT`] on a host without barycentrics: 1 at the
    /// corner's column, 0 elsewhere (each corner is its own vertex here, so
    /// interpolated they are the weights).
    Weights,
    /// [`super::VERTEX_INPUT`]: corner `k`'s value of the vertex output at
    /// byte `at` (none there: 0), the same at all three corners.
    Corner { k: u32, at: Option<u32> },
}

fn build(
    layout: &MeshOut,
    vertex: Vec<SigElement>,
    primitive: &[SigElement],
    ps_inputs: &[SigElement],
) -> Result<Vec<u8>, ShaderError> {
    let unsupported = |what| ShaderError::Unsupported { offset: 0, what };
    if vertex.len() != layout.vertex_at.len() || primitive.len() != layout.primitive_at.len() {
        return Err(unsupported("a mesh layout that does not match its signatures"));
    }
    if vertex.iter().any(|e| e.row.is_none()) {
        return Err(unsupported("a mesh vertex output without a register"));
    }
    if layout.max_primitives == 0 {
        return Err(unsupported("a mesh program without primitives"));
    }
    // The pixel shader's barycentrics stand-ins (a host without
    // barycentrics) are placed past its own inputs, not knowing the mesh
    // program's registers: a vertex output there that the pixel shader does
    // not read is left out, a system value moves past them (as `bary_gs`).
    let ours: Vec<&SigElement> = ps_inputs
        .iter()
        .filter(|e| e.name == super::BARY_INPUT || e.name == super::VERTEX_INPUT)
        .collect();
    if ours.iter().any(|e| e.row.is_none()) {
        return Err(unsupported("a barycentrics stand-in without a register"));
    }
    let taken = |row: Option<u32>| ours.iter().any(|e| e.row == row);
    let mut outputs = Vec::new();
    let mut sources = Vec::new();
    let mut moved = Vec::new();
    for (id, (e, &at)) in vertex.iter().zip(&layout.vertex_at).enumerate() {
        if taken(e.row) {
            if e.kind != 0 {
                moved.push(id);
            }
            continue;
        }
        outputs.push(SigElement {
            usage: (1 << e.cols) - 1,
            ..e.clone()
        });
        sources.push(Source::Out { primitive: false, at });
    }
    // A per-primitive output goes where the pixel shader reads it; one the
    // pixel shader does not read is not drawn with.
    for (p, &at) in primitive.iter().zip(&layout.primitive_at) {
        let Some(e) = ps_inputs
            .iter()
            .find(|e| e.name.eq_ignore_ascii_case(&p.name) && e.index == p.index && e.row.is_some())
        else {
            continue;
        };
        let clash = outputs.iter().any(|o| {
            o.row == e.row && o.col < e.col + p.cols && e.col < o.col + o.cols
        });
        if clash || e.cols < p.cols {
            return Err(unsupported(
                "a per-primitive output whose pixel shader register a vertex output takes",
            ));
        }
        outputs.push(SigElement {
            row: e.row,
            col: e.col,
            interp: 1,
            usage: (1 << p.cols) - 1,
            ..p.clone()
        });
        sources.push(Source::Out { primitive: true, at });
    }
    let used = outputs.iter().chain(ours.iter().copied()).filter_map(|e| e.row).max();
    let mut free = used.map_or(0, |r| r + 1);
    let mut rows: Vec<(Option<u32>, u32)> = Vec::new();
    for id in moved {
        let mut e = vertex[id].clone();
        let row = match rows.iter().find(|(old, _)| *old == e.row) {
            Some(&(_, new)) => new,
            None => {
                rows.push((e.row, free));
                free += 1;
                free - 1
            }
        };
        e.row = Some(row);
        e.usage = (1 << e.cols) - 1;
        outputs.push(e);
        sources.push(Source::Out { primitive: false, at: layout.vertex_at[id] });
    }
    if free > 32 {
        return Err(unsupported("a mesh vertex shader with more than 32 output registers"));
    }
    for e in &ours {
        let source = if e.name == super::BARY_INPUT {
            Source::Weights
        } else {
            let (k, row, col) = super::vertex_source(e.index);
            if !(1..=2).contains(&k) {
                return Err(unsupported("a vertex stand-in of a vertex other than 1 or 2"));
            }
            let at = vertex.iter().zip(&layout.vertex_at).find_map(|(o, &at)| {
                (o.row == Some(row) && col >= o.col && col < o.col + o.cols).then(|| at + 4 * u32::from(col - o.col))
            });
            Source::Corner { k: u32::from(k), at }
        };
        // (as `bary_gs`: a signature has no interpolation mode)
        let interp = if e.name == super::VERTEX_INPUT { 1 } else { 2 };
        outputs.push(SigElement {
            usage: (1 << e.cols) - 1,
            interp,
            ..(*e).clone()
        });
        sources.push(source);
    }

    let mut s = Shader::new(false);
    let system = |name: &str, kind, row| SigElement {
        name: name.into(),
        index: 0,
        kind,
        comp: 5,
        interp: 0,
        row: Some(row),
        col: 0,
        cols: 1,
        usage: 1,
        half: false,
    };
    s.inputs = vec![system("SV_VertexID", 1, 0), system("SV_InstanceID", 2, 1)];
    s.outputs = outputs;
    let b = s.entry;
    let i32t = s.t.i32;
    let k = |s: &mut Shader, v: u32| s.m.i32(v);

    // (vertex id, instance id) -> primitive, corner, group.
    let input = |s: &mut Shader, n: u32| {
        let args = [s.m.i32(n), s.m.i32(0), s.m.i8(0), s.m.undef(i32t)];
        s.op(b, "loadInput.i32", Attr::ReadNone, i32t, 4, &args)
    };
    let (id, group) = (input(&mut s, 0), input(&mut s, 1));
    let three = k(&mut s, 3);
    let prim = s.m.bin(b, Bin::UDiv, id, three);
    let corner = s.m.bin(b, Bin::URem, id, three);
    // A group past the cap wrote to the sink slot: it draws nothing.
    let cap = k(&mut s, layout.groups_cap());
    let past = s.m.cmp(b, super::bitcode::pred::IUGT, group, cap);
    let group = s.op(b, "binary.i32", Attr::ReadNone, i32t, 40, &[group, cap]);
    s.m.tag(group, super::bitcode::Tag::Clamp);

    // Byte offsets in the buffer.
    let slot = k(&mut s, layout.group_bytes());
    let base = s.m.bin(b, Bin::Mul, group, slot);
    let header = k(&mut s, super::shader::MESH_OUT_HEADER);
    let base = s.m.bin(b, Bin::Add, base, header);
    let (first, pstride) = (k(&mut s, layout.primitives_at()), k(&mut s, layout.primitive_stride));
    let prim_at = s.m.bin(b, Bin::Mul, prim, pstride);
    let prim_at = s.m.bin(b, Bin::Add, prim_at, first);
    let prim_at = s.m.bin(b, Bin::Add, prim_at, base);
    let four = k(&mut s, 4);
    let corner_at = s.m.bin(b, Bin::Mul, corner, four);
    let index_at = s.m.bin(b, Bin::Add, prim_at, corner_at);
    let index = load(&mut s, index_at);
    let zero = k(&mut s, 0);
    let index = s.m.select(b, past, zero, index);
    // An index past the vertices (none is written so) reads vertex 0.
    let index = {
        let last = k(&mut s, layout.max_vertices.saturating_sub(1));
        let v = s.op(b, "binary.i32", Attr::ReadNone, i32t, 40, &[index, last]);
        s.m.tag(v, super::bitcode::Tag::Clamp);
        v
    };
    let vstride = k(&mut s, layout.vertex_stride);
    let vertex_at = s.m.bin(b, Bin::Mul, index, vstride);
    let vertex_at = s.m.bin(b, Bin::Add, vertex_at, base);

    let f32t = s.t.f32;
    for (id, source) in sources.iter().enumerate() {
        let (cols, float) = (s.outputs[id].cols, s.outputs[id].comp == 9);
        // Corner `k`'s vertex, for a vertex stand-in.
        let corner_vertex = match *source {
            Source::Corner { k: c, .. } => {
                let off = k(&mut s, 4 * c);
                let at = s.m.bin(b, Bin::Add, prim_at, off);
                let i = load(&mut s, at);
                let i = s.m.select(b, past, zero, i);
                let last = k(&mut s, layout.max_vertices.saturating_sub(1));
                let i = s.op(b, "binary.i32", Attr::ReadNone, i32t, 40, &[i, last]);
                s.m.tag(i, super::bitcode::Tag::Clamp);
                let v = s.m.bin(b, Bin::Mul, i, vstride);
                Some(s.m.bin(b, Bin::Add, v, base))
            }
            _ => None,
        };
        for col in 0..cols {
            let bits = match *source {
                Source::Out { primitive, at } => {
                    let from = if primitive { prim_at } else { vertex_at };
                    let off = k(&mut s, at + 4 * u32::from(col));
                    let at = s.m.bin(b, Bin::Add, from, off);
                    load(&mut s, at)
                }
                Source::Weights => {
                    let c = k(&mut s, u32::from(col));
                    let here = s.m.cmp(b, super::bitcode::pred::EQ, corner, c);
                    let (one, nought) = (s.m.f32(1.0f32.to_bits()), s.m.f32(0));
                    let w = s.m.select(b, here, one, nought);
                    s.m.cast(b, super::bitcode::Cast::Bitcast, w, i32t)
                }
                Source::Corner { at, .. } => match (at, corner_vertex) {
                    (Some(at), Some(v)) => {
                        let off = k(&mut s, at);
                        let at = s.m.bin(b, Bin::Add, v, off);
                        load(&mut s, at)
                    }
                    _ => zero,
                },
            };
            let (name, value) = if float {
                ("storeOutput.f32", s.m.cast(b, super::bitcode::Cast::Bitcast, bits, f32t))
            } else {
                ("storeOutput.i32", bits)
            };
            let args = [k(&mut s, id as u32), k(&mut s, 0), s.m.i8(col), value];
            s.op(b, name, Attr::NoUnwind, s.t.void, 5, &args);
        }
    }
    s.m.terminate(b, Term::Ret);
    Ok(s.finish())
}

/// The dword at byte `at` of [`super::SLOT_MESH_IN`], as an i32.
fn load(s: &mut Shader, at: V) -> V {
    let b = s.entry;
    let handle = s.root_srv_handle(super::SLOT_MESH_IN);
    let (undef, mask, align) = (s.m.undef(s.t.i32), s.m.i8(1), s.m.i32(4));
    let r = s.op(
        b,
        "rawBufferLoad.f32",
        Attr::ReadOnly,
        s.t.res_ret,
        139,
        &[handle, at, undef, mask, align],
    );
    let f = s.m.extract(b, r, 0);
    s.m.cast(b, super::bitcode::Cast::Bitcast, f, s.t.i32)
}
