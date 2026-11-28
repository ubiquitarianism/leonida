//! The test binary's allocator: the system's, telling the tests that count
//! allocations of each block (`modules::apu`'s packet test and
//! `modules::dstorage`'s worker buffer test). A process has one allocator, so
//! they share it. Not with the features under which leonida-gpu installs an
//! allocator of its own.

use std::alloc::{GlobalAlloc, Layout, System};

use crate::modules::{apu, dstorage};

fn note(size: usize) {
    apu::tests::counting::note(size);
    dstorage::tests::watched_blocks::note(size);
}

struct Counting;

// SAFETY: every call goes to the system's allocator unchanged; only counters
// are kept beside it.
unsafe impl GlobalAlloc for Counting {
    unsafe fn alloc(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc(layout) }
    }

    unsafe fn alloc_zeroed(&self, layout: Layout) -> *mut u8 {
        note(layout.size());
        // SAFETY: the caller's contract, passed on.
        unsafe { System.alloc_zeroed(layout) }
    }

    unsafe fn realloc(&self, at: *mut u8, layout: Layout, size: usize) -> *mut u8 {
        note(size);
        // SAFETY: the caller's contract, passed on.
        unsafe { System.realloc(at, layout, size) }
    }

    unsafe fn dealloc(&self, at: *mut u8, layout: Layout) {
        // SAFETY: the caller's contract, passed on.
        unsafe { System.dealloc(at, layout) }
    }
}

#[global_allocator]
static HEAP: Counting = Counting;
