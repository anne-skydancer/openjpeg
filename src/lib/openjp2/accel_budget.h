/* SPDX-License-Identifier: BSD-2-Clause */
#ifndef OPJ_ACCEL_BUDGET_H
#define OPJ_ACCEL_BUDGET_H
#include <stddef.h>
/* Separate aggregate 64 MiB device/pinned-host budgets, shared across backends. */
int opj_accel_reserve(size_t bytes, int host);
size_t opj_accel_usage(int host);
void opj_accel_release(size_t bytes, int host);
#endif
