#include <stdio.h>
#include <math.h>
#include <ap_fixed.h>
#include "hls_stream.h"
#include <iostream>
#include <fstream>


//#define CSIM_DEBUG


typedef ap_uint<8> uint8;
//typedef ap_uint<16> uint16;


#ifdef CSIM_DEBUG
	typedef float FIX_32_4;	//fix point
	typedef float FIX_32_25;	//fix point
	typedef float FIX_FM;	//fix point for feature map
	typedef float FIX_WT;	//fix point for weights
	typedef float FIX_32_16;	//fix point
	typedef float FIX_32_12;	//fix point
	typedef float FIX_16_1;
#else
	typedef ap_fixed<16, 6, AP_TRN_ZERO, AP_SAT> FIX_FM;	//fix point for feature map
	typedef ap_fixed<8,  1, AP_TRN_ZERO, AP_SAT> FIX_WT;	//fix point for weights
	typedef ap_fixed<16, 1, AP_TRN_ZERO, AP_SAT> FIX_16_1;	//fix point for weights
	typedef ap_fixed<32,16, AP_TRN_ZERO, AP_SAT> FIX_32_16;	//fix point
	typedef ap_fixed<32,12, AP_TRN_ZERO, AP_SAT> FIX_32_12;
	typedef ap_fixed<32, 4, AP_TRN_ZERO, AP_SAT> FIX_32_4;	//fix point
	typedef ap_fixed<32,25, AP_TRN_ZERO, AP_SAT> FIX_32_25;	//fix point
#endif


















void mobilenet(uint8 image_in_raw_pad[3][162][322],

				FIX_16_1 fix_conv_weight_1x1_all[405][16][16],
				FIX_16_1 fix_conv_weight_3x3_all[22][16][3][3],
				FIX_16_1 fix_bias_all[67][16],

				FIX_FM DDR_pool_3_out_PL[48][82][162],
				FIX_FM DDR_pool_6_out_PL[96][42][82],

				//FIX_FM DDR_pool_out_PL[96][82][162],

				FIX_FM DDR_buf[36][16][22][42],

				float predict_box[5]

);

void CONV_3x3_group(FIX_FM bottom[16][22][42],
					FIX_FM top[16][22][42],
					FIX_WT weight[16][3][3]);

void CONV_1x1(FIX_FM bottom[16][22][42],
			  FIX_FM top[16][22][42],
			  FIX_WT weights[16][16]);

