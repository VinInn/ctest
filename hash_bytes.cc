// Definition of _Hash_bytes. -*- C++ -*-

// Copyright (C) 2010-2014 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

// This file defines Hash_bytes, a primitive used for defining hash
// functions. Based on public domain MurmurHashUnaligned2, by Austin
// Appleby.  http://murmurhash.googlepages.com/

// This file also defines _Fnv_hash_bytes, another primitive with
// exactly the same interface but using a different hash algorithm,
// Fowler / Noll / Vo (FNV) Hash (type FNV-1a). The Murmur hash
// function apears to be better in both speed and hash quality, and
// FNV is provided primarily for backward compatibility.

#include<string>

namespace
{
  using ub8 = std::size_t;
  constexpr  ub8 less24(char const * k, ub8 len, ub8 a) {
   return  a 
	      + (len<1 ? 0 : (ub8)k[0] )
	      + (len<2 ? 0 :((ub8)k[ 1]<< 8))
	      + (len<3 ? 0 :((ub8)k[ 2]<<16))
	      + (len<4 ? 0 :((ub8)k[ 3]<<24))
	      + (len<5 ? 0 :((ub8)k[4 ]<<32))
	      + (len<6 ? 0 :((ub8)k[ 5]<<40))
	      + (len<7 ? 0 :((ub8)k[ 6]<<48))
	      + (len<8 ? 0 :((ub8)k[ 7]<<56))
     ;
  }

  inline constexpr std::size_t
  unaligned_load(const char* p)
  {    
    return less24(p,sizeof(std::size_t),0);    //__builtin_memcpy(result, p, sizeof(std::size_t) );
  }

  // Loads n bytes, where 1 <= n < 8.
  inline constexpr std::size_t
  load_bytes(const char* p, int n)
  {
    std::size_t result = 0;
    --n;
    do
      result = (result << 8) + static_cast<unsigned char>(p[n]);
    while (--n >= 0);
    return result;
  }

  inline constexpr std::size_t
  shift_mix(std::size_t v)
  { return v ^ (v >> 47);}
}

  // Implementation of Murmur hash for 64-bit size_t.
inline  constexpr size_t
  Hash_bytes(const void* ptr, size_t len, size_t seed)
  {
    constexpr size_t mul = (((size_t) 0xc6a4a793UL) << 32UL)
			      + (size_t) 0x5bd1e995UL;
    const char* const buf = static_cast<const char*>(ptr);

    // Remove the bytes not divisible by the sizeof(size_t).  This
    // allows the main loop to process the data as 64-bit integers.
    const int len_aligned = len & ~0x7;
    const char* const end = buf + len_aligned;
    size_t hash = seed ^ (len * mul);
    for (const char* p = buf; p != end; p += 8)
      {
        size_t n= load_bytes(p,8);
	const size_t data = shift_mix(n * mul) * mul;
	hash ^= data;
	hash *= mul;
      }
    if ((len & 0x7) != 0)
      {
	const size_t data = load_bytes(end, len & 0x7);
	hash ^= data;
	hash *= mul;
      }
    hash = shift_mix(hash) * mul;
    hash = shift_mix(hash);
    return hash;
  }

  // Implementation of FNV hash for 64-bit size_t.
  size_t
  Fnv_hash_bytes(const void* ptr, size_t len, size_t hash)
  {
    const char* cptr = static_cast<const char*>(ptr);
    for (; len; --len)
      {
	hash ^= static_cast<size_t>(*cptr++);
	hash *= static_cast<size_t>(1099511628211ULL);
      }
    return hash;
  }

#include<iostream>
#include<cstring>
int main() {
  constexpr char const * p = "a literal string";
  constexpr size_t n = Hash_bytes(p,strlen(p),0);

  std::cout << n << std::endl;
  return 0;
}
