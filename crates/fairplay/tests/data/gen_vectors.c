/*
 * Prints the reference vectors in playfair_vectors.txt from UxPlay's C
 * FairPlay code, for the differential tests in ../differential.rs.
 *
 * Inputs come from splitmix64, mirrored by the Rust test, so only the C
 * outputs are stored. Regenerate with the UxPlay 1.73.6 sources (for
 * instance `nix build --no-link --print-out-paths nixpkgs#uxplay.src`):
 *
 *   U=<uxplay source>/lib
 *   gcc -O1 -w -I$U gen_vectors.c $U/fairplay_playfair.c $U/playfair/*.c -lm -o gen_vectors
 *   ./gen_vectors > playfair_vectors.txt
 *
 * The C code linked in is UxPlay's (LGPL-2.1-or-later and GPL-3.0); this
 * driver is GPL-3.0-or-later like the crate.
 */
#include <stdint.h>
#include <stdio.h>
#include <string.h>

#include "fairplay.h"

#define CASES 400

static uint64_t state = 0x46504c59;

static uint64_t splitmix64(void) {
  uint64_t z = (state += 0x9e3779b97f4a7c15ULL);
  z = (z ^ (z >> 30)) * 0xbf58476d1ce4e5b9ULL;
  z = (z ^ (z >> 27)) * 0x94d049bb133111ebULL;
  return z ^ (z >> 31);
}

static void fill(unsigned char *out, size_t len) {
  for (size_t i = 0; i < len; i++) out[i] = (unsigned char)(splitmix64() >> 56);
}

static void hex(const char *tag, const unsigned char *bytes, size_t len) {
  printf("%s ", tag);
  for (size_t i = 0; i < len; i++) printf("%02x", bytes[i]);
  printf("\n");
}

static const unsigned char keymsg_header[12] = {'F', 'P', 'L', 'Y', 3, 1, 3, 0, 0, 0, 0, 0x98};
static const unsigned char ekey_header[12] = {'F', 'P', 'L', 'Y', 1, 2, 1, 0, 0, 0, 0, 0x3c};

int main(void) {
  unsigned char reply1[142], reply2[32], key[16];
  unsigned char keymsg[164], ekey[72];

  for (int mode = 0; mode < 4; mode++) {
    unsigned char message1[16] = {'F', 'P', 'L', 'Y', 3, 1, 1, 0, 0, 0, 0, 4, 2, 0, (unsigned char)mode, 0xbb};
    fairplay_t *fp = fairplay_init(NULL);
    if (fairplay_setup(fp, message1, reply1) != 0) return 1;
    hex("setup1", reply1, sizeof reply1);
    fairplay_destroy(fp);
  }

  for (int i = 0; i < CASES; i++) {
    fill(keymsg, sizeof keymsg);
    memcpy(keymsg, keymsg_header, sizeof keymsg_header);
    keymsg[12] = (unsigned char)(i % 4);
    fill(ekey, sizeof ekey);
    memcpy(ekey, ekey_header, sizeof ekey_header);

    fairplay_t *fp = fairplay_init(NULL);
    if (fairplay_handshake(fp, keymsg, reply2) != 0) return 1;
    if (fairplay_decrypt(fp, ekey, key) != 0) return 1;
    hex("setup2", reply2, sizeof reply2);
    hex("key", key, sizeof key);
    fairplay_destroy(fp);
  }
  return 0;
}
