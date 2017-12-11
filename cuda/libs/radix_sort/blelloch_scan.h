/*
 * blelloch_scan.h
 *
 *  Created on: Dec 10, 2017
 *      Author: mathjs
 */

#ifndef BLELLOCH_SCAN_H_
#define BLELLOCH_SCAN_H_

/* Exclusive Scan - Blelloch Scan
 * Performs an exclusive scan in 'd_vec' with size 'vecSize'
 * IMPORTANT: The vector size must be a power of 2, that's a requirement for this algorithm specifically.
 */
void xscan(unsigned int *d_vec, int vecSize, cudaStream_t st);

/* PLEASE define here the identity value and the operator the be used during the scan.
 *
 * OPERATOR    IDENTITY
 * +           0
 * *           1
 * |           0
 * &           1
 */
#define IDENTITY 0
#define OPERATOR +

#endif /* BLELLOCH_SCAN_H_ */
