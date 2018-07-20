

#include "hls_stream.h"
#include <stdio.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls.h"


extern float conv_1_weight_in[16][3][3];
extern float conv_1_bias_in[16];
extern float conv_1_weight_tmp[3][3][3];
extern float conv_1_bias_tmp[3];

extern float conv_2_weight_in[48][16];
extern float conv_2_weight_reorder[3][16][16];
extern float conv_2_weight_tmp[48][3];
extern float conv_2_bias_in[48];

extern float conv_4_weight_in[48][3][3];
extern float conv_4_bias_in[48];

extern float conv_5_weight_in[96][48];
extern float conv_5_bias_in[96];

extern float conv_7_weight_in[96][3][3];
extern float conv_7_bias_in[96];

extern float conv_8_weight_in[192][96];
extern float conv_8_weight_reorder[72][16][16];
extern float conv_8_bias_in[192];

extern float conv_10_weight_in[192][3][3];
extern float conv_10_bias_in[192];

extern float conv_11_weight_in[384][192];
extern float conv_11_weight_reorder[288][16][16];
extern float conv_11_bias_in[384];

extern float conv_12_weight_tmp[10][384];
extern float conv_12_weight_in[16][384];
extern float conv_12_weight_reorder[24][16][16];

extern FIX_WT fix_conv_1_weight_in[16][3][3];
extern FIX_WT fix_conv_1_bias_in[16];

extern FIX_WT fix_conv_2_weight_in[48][16];
extern FIX_WT fix_conv_2_weight_reorder[3][16][16];
extern FIX_WT fix_conv_2_bias_in[48];

extern FIX_WT fix_conv_4_weight_in[48][3][3];
extern FIX_WT fix_conv_4_bias_in[48];

extern FIX_WT fix_conv_5_weight_in[96][48];
extern FIX_WT fix_conv_5_weight_reorder[18][16][16];
extern FIX_WT fix_conv_5_bias_in[96];

extern FIX_WT fix_conv_7_weight_in[96][3][3];
extern FIX_WT fix_conv_7_bias_in[96];

extern FIX_WT fix_conv_8_weight_in[192][96];
extern FIX_WT fix_conv_8_weight_reorder[72][16][16];
extern FIX_WT fix_conv_8_bias_in[192];

extern FIX_WT fix_conv_10_weight_in[192][3][3];
extern FIX_WT fix_conv_10_bias_in[192];

extern FIX_WT fix_conv_11_weight_in[384][192];
extern FIX_WT fix_conv_11_weight_reorder[288][16][16];
extern FIX_WT fix_conv_11_bias_in[384];

extern FIX_WT fix_conv_12_weight_tmp[10][384];
extern FIX_WT fix_conv_12_weight_in[16][384];
extern FIX_WT fix_conv_12_weight_reorder[24][16][16];


//3+18+72+288+24
FIX_WT fix_conv_weight_1x1_all_8[405][16][16];

//1+3+6+12
FIX_WT fix_conv_weight_3x3_all_8[22][16][3][3];

//1+3+3+6+6+12+12+24
FIX_WT fix_bias_all_8[67][16];

extern FIX_16_1 fix_conv_weight_1x1_all[405][16][16];

//1+3+6+12
extern FIX_16_1 fix_conv_weight_3x3_all[22][16][3][3];

//1+3+3+6+6+12+12+24
extern FIX_16_1 fix_bias_all[67][16];


void reorder_weight_fix()
{

    std::ofstream ofs_param_write("params_384_fix.bin", std::ios::out | std::ios::binary);



    //for conv1
    for(int j = 0; j < 3; j++) {
    	for(int k = 0; k < 3; k++) {
    		for(int i = 0; i < 16; i++) {
    			if(i < 3) {
    				conv_1_weight_in[i][j][k] = conv_1_weight_tmp[i][j][k];
    				conv_1_bias_in[i] = conv_1_bias_tmp[i];

    				//for fixed-point data
    				fix_conv_1_weight_in[i][j][k] = (FIX_WT)conv_1_weight_tmp[i][j][k];
    				fix_conv_1_bias_in[i] = (FIX_WT)conv_1_bias_tmp[i];


    			}
    			else {
    				conv_1_weight_in[i][j][k] = 0;
    				conv_1_bias_in[i] = 0;
    				fix_conv_1_weight_in[i][j][k] = 0;
    				fix_conv_1_bias_in[i] = 0;
    			}
    		}
    	}
    }

    //for conv2
    for(int i = 0; i < 48; i++) {
    	fix_conv_2_bias_in[i] = conv_2_bias_in[i];
    	for(int j = 0; j < 16; j++) {
    		if(j < 3) {
    			conv_2_weight_in[i][j] = conv_2_weight_tmp[i][j];
    			fix_conv_2_weight_in[i][j] = (FIX_WT)conv_2_weight_tmp[i][j];
    		}
    		else {
    			conv_2_weight_in[i][j] = 0.0f;
    			fix_conv_2_weight_in[i][j] = 0;
    		}
    	}
    }

    // reorder conv2
    for(int col = 0; col < 3; col++) {
    	for(int row = 0; row < 1; row++) {
    		for(int i = 0; i < 16; i++) {
    			for(int j = 0; j < 16; j++) {
    				fix_conv_2_weight_reorder[col][i][j] = fix_conv_2_weight_in[i + 16*col][j + 16*row];
    			}
    		}
    	}
    }

    //for conv4
    for(int j = 0; j < 48; j++) {
    	fix_conv_4_bias_in[j] = (FIX_WT)conv_4_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_4_weight_in[j][k][i] = (FIX_WT)conv_4_weight_in[j][k][i];
			}
		}
	}

    //for conv5
    for(int j = 0; j < 96; j++) {
		fix_conv_5_bias_in[j] = (FIX_WT)conv_5_bias_in[j];
		for(int k = 0; k < 48; k++) {
			fix_conv_5_weight_in[j][k] = (FIX_WT)conv_5_weight_in[j][k];

		}
	}

    // reorder conv5
    for(int col = 0; col < 6; col++) {
    	for(int row = 0; row < 3; row++) {
    		for(int i = 0; i < 16; i++) {
    			for(int j = 0; j < 16; j++) {
    				fix_conv_5_weight_reorder[col*3 + row][i][j] = fix_conv_5_weight_in[i + col*16][j + row*16];
    			}
    		}
    	}
    }


    //for conv7
	for(int j = 0; j < 96; j++) {
		fix_conv_7_bias_in[j] = (FIX_WT)conv_7_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_7_weight_in[j][k][i] = (FIX_WT)conv_7_weight_in[j][k][i];
			}
		}
	}

    //for conv8
	for(int i = 0; i < 192; i++) {
		fix_conv_8_bias_in[i] = (FIX_WT)conv_8_bias_in[i];
		for(int j = 0; j < 96; j++) {
			fix_conv_8_weight_in[i][j] = (FIX_WT)conv_8_weight_in[i][j];
		}
	}

	//reorder conv8
	for(int col = 0; col < 12; col++ ) {
		for(int row = 0; row < 6; row++) {
			for(int i = 0; i < 16; i++) {
				for(int j = 0; j < 16; j++) {
					fix_conv_8_weight_reorder[col*6 + row][i][j] = fix_conv_8_weight_in[i + col*16][j + row*16];
				}
			}
		}
	}


	//for conv10
	for(int j = 0; j < 192; j++) {
		fix_conv_10_bias_in[j] = (FIX_WT)conv_10_bias_in[j];
		for(int k = 0; k < 3; k++) {
			for(int i = 0; i < 3; i++) {
				fix_conv_10_weight_in[j][k][i] = (FIX_WT)conv_10_weight_in[j][k][i];
			}
		}
	}

	//for conv11
	for(int i = 0; i < 384; i++) {
		fix_conv_11_bias_in[i] = (FIX_WT)conv_11_bias_in[i];
		for(int j = 0; j < 192; j++) {
			fix_conv_11_weight_in[i][j] = (FIX_WT)conv_11_weight_in[i][j];
		}
	}

	//// reorder conv_11_weight
	for(int col = 0; col < 24; col++ ) {
		for(int row = 0; row < 12; row++) {
			for(int i = 0; i < 16; i++) {
				for(int j = 0; j < 16; j++) {
					fix_conv_11_weight_reorder[col*12 + row][i][j] = fix_conv_11_weight_in[i + col*16][j + row*16];
				}
			}
		}
	}

	//for conv12
	for(int ch = 0; ch < 16; ch++ ) {
		for(int i = 0; i < 384; i++) {
			if( ch < 10 ) {
				fix_conv_12_weight_in[ch][i] = (FIX_WT)conv_12_weight_tmp[ch][i];
			}
			else
				fix_conv_12_weight_in[ch][i] = (FIX_WT)0.0;
		}
	}


	//for conv12
	for(int row = 0; row < 24; row++) {
		for(int i = 0; i < 16; i++) {
			for(int j = 0; j < 16; j++) {
				fix_conv_12_weight_reorder[row][i][j] = fix_conv_12_weight_in[i][j + row*16];
			}
		}
	}


	//////////// put all reordered weights together

	// copy conv_1 to conv_weight_3x3_all
	int index_3x3 = 0;
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++)
				fix_conv_weight_3x3_all[index_3x3][i][j][k] = fix_conv_1_weight_in[i][j][k];
		}
	}

	// copy conv_4 to conv_weight_3x3_all
	for(int i = 0; i < 48; i++) {
		if( i % 16 == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%16][j][k] = fix_conv_4_weight_in[i][j][k];
			}
		}
	}


	// copy conv_7 to conv_weight_3x3_all
	for(int i = 0; i < 96; i++) {
		if( i % 16 == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%16][j][k] = fix_conv_7_weight_in[i][j][k];
			}
		}
	}

	// copy conv_10 to conv_weight_3x3_all
	for(int i = 0; i < 192; i++) {
		if( i % 16 == 0 )
			index_3x3++;

		for(int j = 0; j < 3; j++) {
			for(int k = 0; k < 3; k++) {
				fix_conv_weight_3x3_all[index_3x3][i%16][j][k] = fix_conv_10_weight_in[i][j][k];
			}
		}
	}

	// copy conv_2_reorder to conv_weight_1x1_all
	int index_1x1 = -1;
	for(int i = 0; i < 3; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_2_weight_reorder[i][j][k];
			}
		}
	}


	// copy conv_5_reorder to conv_weight_1x1_all
	for(int i = 0; i < 18; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_5_weight_reorder[i][j][k];
			}
		}
	}


	// copy conv_8_reorder to conv_weight_1x1_all
	for(int i = 0; i < 72; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_8_weight_reorder[i][j][k];
			}
		}
	}

	// copy conv_11_reorder to conv_weight_1x1_all
	for(int i = 0; i < 288; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_11_weight_reorder[i][j][k];
			}
		}
	}

	// copy conv_12_reorder to conv_weight_1x1_all
	for(int i = 0; i < 24; i++) {
		index_1x1++;

		for(int j = 0; j < 16; j++) {
			for(int k = 0; k < 16; k++) {
				fix_conv_weight_1x1_all[index_1x1][j][k] = fix_conv_12_weight_reorder[i][j][k];
			}
		}
	}


	// put all bias into one array fix_bias_all[93][16][16]
	// copy conv_1_bias to fix_bias_all
	int index_bias = 0;
	for(int i = 0; i < 16; i++) {
		fix_bias_all[index_bias][i] = fix_conv_1_bias_in[i];
	}

	// copy conv_2_bias to fix_bias_all
	for(int ch = 0; ch < 3; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_2_bias_in[ch * 16 + i];
		}
	}


	// copy conv_4_bias to fix_bias_all
	for(int ch = 0; ch < 3; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_4_bias_in[ch * 16 + i];
		}
	}

	// copy conv_5_bias to fix_bias_all
	for(int ch = 0; ch < 6; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_5_bias_in[ch * 16 + i];
		}
	}


	// copy conv_7_bias to fix_bias_all
	for(int ch = 0; ch < 6; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_7_bias_in[ch * 16 + i];
		}
	}

	// copy conv_8_bias to fix_bias_all
	for(int ch = 0; ch < 12; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_8_bias_in[ch * 16 + i];
		}
	}


	// copy conv_10_bias to fix_bias_all
	for(int ch = 0; ch < 12; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_10_bias_in[ch * 16 + i];
		}
	}

	// copy conv_11_bias to fix_bias_all
	for(int ch = 0; ch < 24; ch++) {
		index_bias++;

		for(int i = 0; i < 16; i++) {
			fix_bias_all[index_bias][i] = fix_conv_11_bias_in[ch * 16 + i];
		}
	}

	printf("index_1x1: %d\n", index_1x1);
	printf("index_3x3: %d\n", index_3x3);
	printf("index_bias: %d\n", index_bias);


    // write conv_1x1 weights into params_fix_384.bin
    ofs_param_write.write((char*)fix_conv_weight_1x1_all, 405*16*16*sizeof(FIX_16_1));

    // write conv_3x3 into params_fix_384.bin
    ofs_param_write.write((char*)fix_conv_weight_3x3_all, 22*16*3*3*sizeof(FIX_16_1));

    // write bias_all into params_fix_384.bin
    ofs_param_write.write((char*)fix_bias_all, 67*16*sizeof(FIX_16_1));

    ofs_param_write.close();

}

