# XG texture layout references

The thin surface size, mip order and packed mip tail rules in
`xg_xs_layout.h` use AMD addrlib's GFX10 implementation. The
`k64kbStandard` and `k4kbStandard3D` equations come from AMD's RB+ tables.
The source pin is [Mesa 26.2.0 addrlib][amd], in
`src/gfx10/gfx10addrlib.cpp` and `src/gfx10/gfx10SwizzlePattern.h`.
The existing Leonida detiler supplies the 2D 4 KiB standard and 64 KiB
display equations.

The Series descriptor and texture computer ABI and the BC7 fixture
were cross-checked with [VIctory, commit 2849477][victory], GPL-3.0-or-later,
in `crates/victory-hle/src/modules/xg.rs`. The output structure sizes were
also checked against [xbox-emulator, commit 410e01f][xbox-emulator],
GPL-3.0, in `crates/compat/xg_xs/src/{computer,layout}.rs`.

xcl (GPL-2.0-only) provided evidence about GFX10 mip ordering, compressed
block units and the slot 10 contract. TuxBox (Apache-2.0) provided
published differential coverage for tile shapes and mode selection.
Both were references for behavior. The swizzle equations used here
come from the original MIT-licensed AMD sources.

[amd]: https://gitlab.freedesktop.org/mesa/mesa/-/tree/mesa-26.2.0/src/amd/addrlib
[victory]: https://github.com/Marvisak/VIctory/tree/2849477
[xbox-emulator]: https://github.com/omonkeyz/xbox-emulator/tree/410e01f

## AMD addrlib

Copyright (C) 2007-2026 Advanced Micro Devices, Inc. All rights reserved.
Copyright (C) 2007-2024 Advanced Micro Devices, Inc. All rights reserved.

SPDX-License-Identifier: MIT

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
