sgBox % L64Hash("12345678")
MsgBox % L128Hash("12345678")

L64Hash(x) {                        ; 64-bit generalized LFSR hash of string x
   Local i, R = 0
   LHashInit()                      ; 1st time set LHASH0..LHAS256 global table
   Loop Parse, x
   {
      i := (R >> 56) & 255          ; dynamic vars are global
      R := (R << 8) + Asc(A_LoopField) ^ LHASH%i%
   }
   Return Hex8(R>>32) . Hex8(R)
}

L128Hash(x) {                       ; 128-bit generalized LFSR hash of string x
   Local i, S = 0, R = -1
   LHashInit()                      ; 1st time set LHASH0..LHAS256 global table
   Loop Parse, x
   {
      i := (R >> 56) & 255          ; dynamic vars are global
      R := (R << 8) + Asc(A_LoopField) ^ LHASH%i%
      i := (S >> 56) & 255
      S := (S << 8) + Asc(A_LoopField) - LHASH%i%
   }
   Return Hex8(R>>32) . Hex8(R) . Hex8(S>>32) . Hex8(S)
}

Hex8(i) {                           ; integer -> LS 8 hex digits
   SetFormat Integer, Hex
   i:= 0x100000000 | i & 0xFFFFFFFF ; mask LS word, set bit32 for leading 0's --> hex
   SetFormat Integer, D
   Return SubStr(i,-7)              ; 8 LS digits = 32 unsigned bits
}

LHashInit() {                       ; build pseudorandom substitution table
   Local i, u = 0, v = 0
   If LHASH0=
      Loop 256 {
         i := A_Index - 1
         TEA(u,v, 1,22,333,4444, 8) ; <- to be portable, no Random()
         LHASH%i% := (u<<32) | v
      }
}
                                    ; [y,z] = 64-bit I/0 block, [k0,k1,k2,k3] = 128-bit key
TEA(ByRef y,ByRef z, k0,k1,k2,k3, n = 32) { ; n = #Rounds
   s := 0, d := 0x9E3779B9
   Loop %n% {                       ; standard = 32, 8 for speed
      k := "k" . s & 3              ; indexing the key
      y := 0xFFFFFFFF & (y + ((z << 4 ^ z >> 5) + z  ^  s + %k%))
      s := 0xFFFFFFFF & (s + d)     ; simulate 32 bit operations
      k := "k" . s >> 11 & 3
      z := 0xFFFFFFFF & (z + ((y << 4 ^ y >> 5) + y  ^  s + %k%))
   }
}
