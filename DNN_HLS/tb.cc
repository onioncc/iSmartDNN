
#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls.h"

float image[3][160][320];

float conv_1_weight_in[16][3][3];
float conv_1_bias_in[16];
float conv_1_weight_tmp[3][3][3];
float conv_1_bias_tmp[3];

float conv_2_weight_in[48][16];
float conv_2_weight_reorder[3][16][16];
float conv_2_weight_tmp[48][3];
float conv_2_bias_in[48];

float conv_4_weight_in[48][3][3];
float conv_4_bias_in[48];

float conv_5_weight_in[96][48];
float conv_5_bias_in[96];

float conv_7_weight_in[96][3][3];
float conv_7_bias_in[96];

float conv_8_weight_in[192][96];
float conv_8_weight_reorder[72][16][16];
float conv_8_bias_in[192];

float conv_10_weight_in[192][3][3];
float conv_10_bias_in[192];

float conv_11_weight_in[384][192];
float conv_11_weight_reorder[288][16][16];
float conv_11_bias_in[384];

float conv_12_weight_tmp[10][384];
float conv_12_weight_in[16][384];
float conv_12_weight_reorder[24][16][16];

FIX_WT fix_conv_1_weight_in[16][3][3];
FIX_WT fix_conv_1_bias_in[16];

FIX_WT fix_conv_2_weight_in[48][16];
FIX_WT fix_conv_2_weight_reorder[3][16][16];
FIX_WT fix_conv_2_bias_in[48];

FIX_WT fix_conv_4_weight_in[48][3][3];
FIX_WT fix_conv_4_bias_in[48];

FIX_WT fix_conv_5_weight_in[96][48];
FIX_WT fix_conv_5_weight_reorder[18][16][16];
FIX_WT fix_conv_5_bias_in[96];

FIX_WT fix_conv_7_weight_in[96][3][3];
FIX_WT fix_conv_7_bias_in[96];

FIX_WT fix_conv_8_weight_in[192][96];
FIX_WT fix_conv_8_weight_reorder[72][16][16];
FIX_WT fix_conv_8_bias_in[192];

FIX_WT fix_conv_10_weight_in[192][3][3];
FIX_WT fix_conv_10_bias_in[192];

FIX_WT fix_conv_11_weight_in[384][192];
FIX_WT fix_conv_11_weight_reorder[288][16][16];
FIX_WT fix_conv_11_bias_in[384];

FIX_WT fix_conv_12_weight_tmp[10][384];
FIX_WT fix_conv_12_weight_in[16][384];
FIX_WT fix_conv_12_weight_reorder[24][16][16];


//3+18+72+288+24
FIX_16_1 fix_conv_weight_1x1_all[405][16][16];

//1+3+6+12
FIX_16_1 fix_conv_weight_3x3_all[22][16][3][3];

//1+3+3+6+6+12+12+24
FIX_16_1 fix_bias_all[67][16];


FIX_FM DDR_pool_3_out_PL[48][82][162];		/// DDR storage for pool3 output with padding
FIX_FM DDR_pool_6_out_PL[96][42][82];		/// DDR storage for pool6 output with padding

FIX_FM DDR_pool_out_PL[96][82][162];	/// DDR Storage for pooling layers' output


uint8  fix_image_raw[3][160][320];	// 0~255 RGB raw data
uint8  fix_image_raw_pad[3][162][322];	// 0~255 RGB raw data


FIX_FM DDR_buf[36][16][22][42];


FIX_32_25 my_exp_fix(FIX_FM input);

void golden_model();
void produce_weight_bias();
void reorder_weight_fix();



int test_one_frame( char* filename )
{
    std::ifstream ifs_param("params_384_320_160_v2.bin", std::ios::in | std::ios::binary);

    ///////////// Prepare Image //////////////////////
    std::ifstream ifs_image_raw(filename, std::ios::in | std::ios::binary);
    ifs_image_raw.read((char*)(**fix_image_raw), 3*160*320*sizeof(uint8));


    ///////////////// PADDING FOR RAW IMAGE ///////////
    for(int i = 0; i < 3; i++) {
		for(int j = 0; j < 162; j++) {
			for(int k = 0; k < 322; k++) {
				if(j==0 || k==0 || j==177 || k==321) {
					fix_image_raw_pad[i][j][k] = 127;
				}
				else {
					fix_image_raw_pad[i][j][k] = fix_image_raw[i][j-1][k-1];
				}
			}
		}
    }

    ///////////////// IMAGE NORM ///////////////////
	for(int j = 0; j < 160; j++) {
		for(int k = 0; k < 320; k++) {
			image[0][j][k] = (((fix_image_raw[0][j][k].to_int()/255.0)-0.5)/0.25);
			image[1][j][k] = (((fix_image_raw[1][j][k].to_int()/255.0)-0.5)/0.25);
			image[2][j][k] = (((fix_image_raw[2][j][k].to_int()/255.0)-0.5)/0.25);
		}
	}


    //std::cout << image[0][0][0] << " " << image[1][0][0] << " " << image[2][0][0] << std::endl;


    ///////////// Read Weights ///////////////////////
    ifs_param.read((char*)(**conv_1_weight_tmp), 3*3*3*sizeof(float));
    ifs_param.read((char*)conv_1_bias_tmp, 3*sizeof(float));
    ifs_param.read((char*)(*conv_2_weight_tmp), 48*3*sizeof(float));
    ifs_param.read((char*)conv_2_bias_in, 48*sizeof(float));
    ifs_param.read((char*)(**conv_4_weight_in), 48*3*3*sizeof(float));
    ifs_param.read((char*)conv_4_bias_in, 48*sizeof(float));
    ifs_param.read((char*)(*conv_5_weight_in), 96*48*sizeof(float));
    ifs_param.read((char*)conv_5_bias_in, 96*sizeof(float));
    ifs_param.read((char*)(**conv_7_weight_in), 96*3*3*sizeof(float));
    ifs_param.read((char*)conv_7_bias_in, 96*sizeof(float));
    ifs_param.read((char*)(*conv_8_weight_in), 192*96*sizeof(float));
    ifs_param.read((char*)conv_8_bias_in, 192*sizeof(float));
    ifs_param.read((char*)(**conv_10_weight_in), 192*3*3*sizeof(float));
    ifs_param.read((char*)conv_10_bias_in, 192*sizeof(float));
    ifs_param.read((char*)(*conv_11_weight_in), 384*192*sizeof(float));
    ifs_param.read((char*)conv_11_bias_in, 384*sizeof(float));
    ifs_param.read((char*)(*conv_12_weight_tmp), 10*384*sizeof(float));
    ifs_param.close();


    /////// GOLDEN MODEL ///////////
    printf("Computing Golden Model...\n");
    golden_model();

    reorder_weight_fix();


    float predict_box[5];

    mobilenet(fix_image_raw_pad,

    		fix_conv_weight_1x1_all,
    		fix_conv_weight_3x3_all,
			fix_bias_all,

			DDR_pool_3_out_PL,
			DDR_pool_6_out_PL,

			//DDR_pool_out_PL,

			DDR_buf,

			predict_box

			);


    return 0;
}



int main()
{


	printf("1.bin\n");
	test_one_frame( "1.bin" );

	/*printf("\n\n1.bin\n");
	test_one_frame( "1.bin" );

	printf("\n\n2.bin\n");
	test_one_frame( "2.bin" );

	printf("\n\n3.bin\n");
	test_one_frame( "3.bin" );

	printf("\n\n4.bin\n");
	test_one_frame( "4.bin" );

	printf("\n\n5.bin\n");
	test_one_frame( "5.bin" );

	printf("\n\n6.bin\n");
	test_one_frame( "6.bin" );

	printf("\n\n7.bin\n");
	test_one_frame( "7.bin" );

	printf("\n\n8.bin\n");
	test_one_frame( "8.bin" );

	printf("\n\n9.bin\n");
	test_one_frame( "9.bin" );

	printf("\n\n10.bin\n");
	test_one_frame( "10.bin" );*/


	return 0;

}





