#include "bipbuffer.h"

#include <stdlib.h>

#include "pico/stdlib.h"

struct _bipbuffer_t
{
    uint32_t size;
    uint32_t write;
    uint32_t read;
    uint32_t last;
    uint32_t reserve;
    bool read_in_progress;
    bool write_in_progress;
    bool already_split;
    uint8_t buf[];
};

struct _bipbuffer_reader_t
{
    bipbuffer_t *bbuf;
};

struct _bipbuffer_writer_t
{
    bipbuffer_t *bbuf;
};

static inline uint32_t bipbuf_sizeof(const uint32_t size)
{
    return sizeof(bipbuffer_t) + size;
}

bipbuffer_t *bipbuffer_new(const uint32_t size)
{
    bipbuffer_t *self = calloc(1, bipbuf_sizeof(size));
    if (!self)
        return NULL;
    self->size = size;
    return self;
}

bipbuffer_rw_pair_t bipbuffer_split(bipbuffer_t *const self)
{
    bool ret;
    const bool val = true;

    __atomic_exchange(&self->already_split, &val, &ret, __ATOMIC_ACQ_REL);
    if (ret)
    {
        return (bipbuffer_rw_pair_t){
            .reader = NULL,
            .writer = NULL,
        };
    }

    bipbuffer_reader_t *reader = calloc(1, sizeof(bipbuffer_reader_t));
    bipbuffer_writer_t *writer = calloc(1, sizeof(bipbuffer_writer_t));

    reader->bbuf = self;
    writer->bbuf = self;

    return (bipbuffer_rw_pair_t){.reader = reader,
                                 .writer = writer};
}

bipbuffer_wgrant_t bipbuffer_writer_grant_exact(bipbuffer_writer_t *self, uint32_t size)
{
    bool ret;
    const bool val = true;
    bipbuffer_t *const inner = self->bbuf;

    __atomic_exchange(&inner->write_in_progress, &val, &ret, __ATOMIC_ACQ_REL);
    if (ret)
    {
        return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
    }

    const size_t write = __atomic_load_n(&inner->write, __ATOMIC_ACQUIRE);
    const size_t read = __atomic_load_n(&inner->read, __ATOMIC_ACQUIRE);
    const size_t max = inner->size;
    const bool already_inverted = write < read;
    size_t start;

    if (already_inverted)
    {
        if ((write + size) < read)
        {
            start = write;
        }
        else
        {
            __atomic_store_n(&inner->write_in_progress, false, __ATOMIC_RELEASE);
            return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
        }
    }
    else
    {
        if ((write + size) <= max)
        {
            start = write;
        }
        else
        {
            if (size < read)
            {
                start = 0;
            }
            else
            {
                __atomic_store_n(&inner->write_in_progress, false, __ATOMIC_RELEASE);
                return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
            }
        }
    }
    __atomic_store_n(&inner->reserve, start + size, __ATOMIC_RELEASE);

    return (bipbuffer_wgrant_t){.data = inner->buf + start, .len = size};
}

bipbuffer_wgrant_t bipbuffer_writer_grant_max(bipbuffer_writer_t *self, uint32_t size)
{
    bool ret;
    const bool val = true;
    bipbuffer_t *const inner = self->bbuf;

    __atomic_exchange(&inner->write_in_progress, &val, &ret, __ATOMIC_ACQ_REL);
    if (ret)
    {
        return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
    }

    const size_t write = __atomic_load_n(&inner->write, __ATOMIC_ACQUIRE);
    const size_t read = __atomic_load_n(&inner->read, __ATOMIC_ACQUIRE);
    const size_t max = inner->size;
    const bool already_inverted = write < read;
    size_t start;

    if (already_inverted)
    {
        const size_t remain = read - write - 1;
        if (remain != 0)
        {
            size = remain < size ? remain : size;
            start = write;
        }
        else
        {
            __atomic_store_n(&inner->write_in_progress, false, __ATOMIC_RELEASE);
            return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
        }
    }
    else
    {
        if (write != max)
        {
            size = (max - write) < size ? (max - write) : size;
            start = write;
        }
        else
        {
            if (read > 1)
            {
                size = (read - 1) < size ? (read - 1) : size;
                start = 0;
            }
            else
            {
                __atomic_store_n(&inner->write_in_progress, false, __ATOMIC_RELEASE);
                return (bipbuffer_wgrant_t){.data = NULL, .len = 0};
            }
        }
    }
    __atomic_store_n(&inner->reserve, start + size, __ATOMIC_RELEASE);
    return (bipbuffer_wgrant_t){.data = inner->buf + start, .len = size};
}

void bipbuffer_writer_commit(bipbuffer_writer_t *self, bipbuffer_wgrant_t const *const grant, uint32_t used)
{
    bipbuffer_t *const inner = self->bbuf;
    if (!__atomic_load_n(&inner->write_in_progress, __ATOMIC_ACQUIRE))
    {
        return;
    }
    const size_t len = grant->len;
    used = len < used ? len : used;

    const size_t write = __atomic_load_n(&inner->write, __ATOMIC_ACQUIRE);
    __atomic_fetch_sub(&inner->reserve, len - used, __ATOMIC_ACQ_REL);

    const size_t max = inner->size;
    const size_t last = __atomic_load_n(&inner->last, __ATOMIC_ACQUIRE);
    const size_t new_write = __atomic_load_n(&inner->reserve, __ATOMIC_ACQUIRE);

    if ((new_write < write) && (write != max))
    {
        __atomic_store_n(&inner->last, write, __ATOMIC_RELEASE);
    }
    else if (new_write > last)
    {
        __atomic_store_n(&inner->last, max, __ATOMIC_RELEASE);
    }
    __atomic_store_n(&inner->write, new_write, __ATOMIC_RELEASE);
    __atomic_store_n(&inner->write_in_progress, false, __ATOMIC_RELEASE);
}

bipbuffer_rgrant_t bipbuffer_reader_read(bipbuffer_reader_t *self)
{
    bool ret;
    const bool val = true;
    bipbuffer_t *const inner = self->bbuf;

    __atomic_exchange(&inner->read_in_progress, &val, &ret, __ATOMIC_ACQ_REL);
    if (ret)
    {
        return (bipbuffer_rgrant_t){.data = NULL, .len = 0};
    }

    const size_t write = __atomic_load_n(&inner->write, __ATOMIC_ACQUIRE);
    const size_t last = __atomic_load_n(&inner->last, __ATOMIC_ACQUIRE);
    size_t read = __atomic_load_n(&inner->read, __ATOMIC_ACQUIRE);

    if ((read == last) && (write < read))
    {
        read = 0;
        __atomic_store_n(&inner->read, 0, __ATOMIC_RELEASE);
    }

    size_t size;
    if (write < read)
    {
        size = last;
    }
    else
    {
        size = write;
    }
    size -= read;

    if (size == 0)
    {
        __atomic_store_n(&inner->read_in_progress, false, __ATOMIC_RELEASE);
        return (bipbuffer_rgrant_t){.data = NULL, .len = 0};
    }

    return (bipbuffer_rgrant_t){.data = inner->buf + read, .len = size};
}

void bipbuffer_reader_release(bipbuffer_reader_t *self, bipbuffer_rgrant_t const *const grant, uint32_t used)
{
    used = grant->len < used ? grant->len : used;
    bipbuffer_t *const inner = self->bbuf;

    if (!__atomic_load_n(&inner->read_in_progress, __ATOMIC_ACQUIRE))
    {
        return;
    }

    hard_assert(used <= grant->len);

    __atomic_fetch_add(&inner->read, used, __ATOMIC_RELEASE);
    __atomic_store_n(&inner->read_in_progress, false, __ATOMIC_RELEASE);
}

bool bipbuffer_release(bipbuffer_t *const self, bipbuffer_reader_t *const rd, bipbuffer_writer_t *const wr)
{
    const bool our_rd = rd->bbuf == self;
    const bool our_wr = wr->bbuf == self;

    if (!(our_rd && our_wr))
    {
        return false;
    }

    const bool read_in_progress = __atomic_load_n(&self->read_in_progress, __ATOMIC_ACQUIRE);
    const bool write_in_progress = __atomic_load_n(&self->write_in_progress, __ATOMIC_ACQUIRE);

    if (read_in_progress || write_in_progress)
    {
        return false;
    }

    free(rd);
    free(wr);
    free(self);

    return true;
}
