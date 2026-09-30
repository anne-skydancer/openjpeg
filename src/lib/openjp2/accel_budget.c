/* SPDX-License-Identifier: BSD-2-Clause */
#include "accel_budget.h"
#include <assert.h>
#ifdef _WIN32
#include <windows.h>
static SRWLOCK lock = SRWLOCK_INIT;
#define LOCK() AcquireSRWLockExclusive(&lock)
#define UNLOCK() ReleaseSRWLockExclusive(&lock)
#else
#include <pthread.h>
static pthread_mutex_t lock = PTHREAD_MUTEX_INITIALIZER;
#define LOCK() pthread_mutex_lock(&lock)
#define UNLOCK() pthread_mutex_unlock(&lock)
#endif
static size_t used[2];
int opj_accel_reserve(size_t bytes, int host)
{
    int ok;
    LOCK();
    ok = bytes <= 64u*1024u*1024u-used[!!host];
    if(ok) used[!!host] += bytes;
    UNLOCK();
    return ok;
}
void opj_accel_release(size_t bytes, int host)
{
    LOCK(); assert(bytes <= used[!!host]); used[!!host] -= bytes; UNLOCK();
}

size_t opj_accel_usage(int host)
{
    size_t bytes;
    LOCK();bytes=used[!!host];UNLOCK();return bytes;
}
