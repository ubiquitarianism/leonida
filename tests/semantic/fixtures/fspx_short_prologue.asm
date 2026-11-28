; Two synthetic VOP3 NOPs with SRC2 selecting a literal. Each consumes 12 bytes.
; The second literal resembles s_endpgm. Only the last instruction ends the program.
.long 0xd5800011, 0x03fe0102, 0x12345678
.long 0xd5800022, 0x03fe0501, 0xbf810000
s_nop 0
s_endpgm
