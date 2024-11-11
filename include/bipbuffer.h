#pragma once

#include <stddef.h>
#include <stdint.h>
#include <stdbool.h>

typedef struct _bipbuffer_t bipbuffer_t;
typedef struct _bipbuffer_reader_t bipbuffer_reader_t;
typedef struct _bipbuffer_writer_t bipbuffer_writer_t;

typedef struct
{
    bipbuffer_reader_t *reader;
    bipbuffer_writer_t *writer;
} bipbuffer_rw_pair_t;

typedef struct
{
    uint8_t *const data;
    const uint32_t len;
} bipbuffer_wgrant_t;

typedef struct
{
    uint8_t *const data;
    const uint32_t len;
} bipbuffer_rgrant_t;

bipbuffer_t *bipbuffer_new(const uint32_t size);
bipbuffer_rw_pair_t bipbuffer_split(bipbuffer_t *const self);

bipbuffer_wgrant_t bipbuffer_writer_grant_exact(bipbuffer_writer_t *self, uint32_t size);
bipbuffer_wgrant_t bipbuffer_writer_grant_max(bipbuffer_writer_t *self, uint32_t size);
void bipbuffer_writer_commit(bipbuffer_writer_t *self, bipbuffer_wgrant_t const * const grant, uint32_t used);

bipbuffer_rgrant_t bipbuffer_reader_read(bipbuffer_reader_t *self);
void bipbuffer_reader_release(bipbuffer_reader_t *self, bipbuffer_rgrant_t const * const grant, uint32_t used);

bool bipbuffer_release(bipbuffer_t *const self, bipbuffer_reader_t *const rd, bipbuffer_writer_t *const wr);
