
#include "hls_stream.h"
#include <stdio.h>
#include <stdlib.h>
#include <iostream>
#include <fstream>
#include <math.h>
#include <ap_fixed.h>
#include "net_hls.h"


#define EPSILON	1e-004
//#define EPSILON	0.5f

extern float image[3][160][320];

extern float conv_1_weight_tmp[3][3][3];
extern float conv_1_bias_tmp[3];

extern float conv_2_weight_tmp[48][3];
extern float conv_2_bias_in[48];

extern float conv_4_weight_in[48][3][3];
extern float conv_4_bias_in[48];

extern float conv_5_weight_in[96][48];
extern float conv_5_bias_in[96];

extern float conv_7_weight_in[96][3][3];
extern float conv_7_bias_in[96];

extern float conv_8_weight_in[192][96];
extern float conv_8_bias_in[192];

extern float conv_10_weight_in[192][3][3];
extern float conv_10_bias_in[192];

extern float conv_11_weight_in[384][192];
extern float conv_11_bias_in[384];

extern float conv_12_weight_tmp[10][384];




float conv_1_out_PL[3][160][320];
float conv_2_out_PL[48][160][320];
float pool_3_out_PL[48][80][160];
float conv_4_out_PL[48][80][160];
float conv_5_out_PL[96][80][160];
float pool_6_out_PL[96][40][80];
float conv_7_out_PL[96][40][80];
float conv_8_out_PL[192][40][80];
float pool_9_out_PL[192][20][40];
float conv_10_out_PL[192][20][40];
float conv_11_out_PL[384][20][40];
float conv_12_out_PL[16][20][40];
extern FIX_FM DDR_pool_3_out_PL[48][82][162];


float conv_1_out[3][160][320];
float conv_2_out[48][160][320];
float pool_3_out[48][80][160];
float conv_4_out[48][80][160];
float conv_5_out[96][80][160];
float pool_6_out[96][40][80];
float conv_7_out[96][40][80];
float conv_8_out[192][40][80];
float pool_9_out[192][20][40];
float conv_10_out[192][20][40];
float conv_11_out[384][20][40];
float conv_12_out[10][20][40];


using namespace std;

FILE* fo;

float max_4(float a1, float a2, float a3, float a4)
{
    float tmp1, tmp2;

    if(a1 > a2) tmp1 = a1; else tmp1 = a2;
    if(a3 > a4) tmp2 = a3; else tmp2 = a4;
    if(tmp1 > tmp2) return tmp1; else return tmp2;
}



void conv_1(
            float input[3][160][320],
            float weight[3][3][3],
            float bias[3],
            float output[3][160][320]
            )
{
    //cout << "conv_1..." << endl;

    for(int co = 0; co < 3; co++) {
        for(int h = 0; h < 160; h++) {
            for(int w = 0; w < 320; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 160 && w+n-1 < 320) ? input[co][h+m-1][w+n-1] : 0);


                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_1_out", "w");
    for(int i = 0; i < 3; i++) {
        for(int j = 0; j < 160; j++) {
            for(int k = 0; k < 320; k ++) {
                fprintf(fo, "conv_1_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void conv_2(
            float input[3][160][320],
            float weight[48][3],
            float bias[48],
            float output[48][160][320]
            )
{
    //cout << "conv_2..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 169; h++) {
            for(int w = 0; w < 320; w++) {
                float sum = 0;

                for(int ci = 0; ci < 3; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_2_out", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 160; j++) {
            for(int k = 0; k < 320; k ++) {
                fprintf(fo, "conv_2_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void max_pool_3(
                   float input[48][160][320],
                   float output[48][80][160]
                   )
{
    //cout << "max_pool_3..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fo = fopen("max_pool_3_out", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "max_pool_3_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void conv_4(
            float input[48][80][160],
            float weight[48][3][3],
            float bias[48],
            float output[48][80][160]
            )
{
    //cout << "conv_4..." << endl;

    for(int co = 0; co < 48; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {
                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 80 && w+n-1 < 160) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }


    fo = fopen("conv_4_out", "w");
    for(int i = 0; i < 48; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "conv_4_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void conv_5(
            float input[48][80][160],
            float weight[96][48],
            float bias[96],
            float output[96][80][160]
            )
{
    //cout << "conv_5..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 80; h++) {
            for(int w = 0; w < 160; w++) {
                float sum = 0;

                for(int ci = 0; ci < 48; ci++ ) {
                	/*if(co==0 && h==0 && w==20)
						printf("%f * %f = %f\n", weight[co][ci], input[ci][h][w], sum);*/

                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }


    fo = fopen("conv_5_out", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 80; j++) {
            for(int k = 0; k < 160; k ++) {
                fprintf(fo, "conv_5_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void max_pool_6(
                float input[96][80][160],
                float output[96][40][80]
                )
{
    //cout << "max_pool_6..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }


    fo = fopen("max_pool_6_out", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "max_pool_6_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void conv_7(
            float input[96][40][80],
            float weight[96][3][3],
            float bias[96],
            float output[96][40][80]
            )
{
    //cout << "conv_7..." << endl;

    for(int co = 0; co < 96; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 40 && w+n-1 < 80) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_7_out", "w");
    for(int i = 0; i < 96; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "conv_7_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void conv_8(
            float input[96][40][80],
            float weight[192][96],
            float bias[192],
            float output[192][40][80]
            )
{
    //cout << "conv_8..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 40; h++) {
            for(int w = 0; w < 80; w++) {
                float sum = 0;

                for(int ci = 0; ci < 96; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }

                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_8_out", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 40; j++) {
            for(int k = 0; k < 80; k ++) {
                fprintf(fo, "conv_8_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}


void max_pool_9(
                float input[192][40][80],
                float output[192][20][40]
                )
{
    //cout << "max_pool_9..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {

                output[co][h][w] = max_4(
                                        input[co][h*2][w*2],
                                        input[co][h*2+1][w*2],
                                        input[co][h*2][w*2+1],
                                        input[co][h*2+1][w*2+1]
                                        );
            }
        }
    }

    fo = fopen("max_pool_9_out", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "max_pool_9_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}



void conv_10(
            float input[192][20][40],
            float weight[192][3][3],
            float bias[192],
            float output[192][20][40]
            )
{
    //cout << "conv_10..." << endl;

    for(int co = 0; co < 192; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int m = 0; m < 3; m++) {
                    for(int n = 0; n < 3; n++) {

                        sum += weight[co][m][n] *
                                (( h+m-1 >= 0 && w+n-1 >= 0 && h+m-1 < 20 && w+n-1 < 40) ? input[co][h+m-1][w+n-1] : 0);
                    }
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_10_out", "w");
    for(int i = 0; i < 192; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "conv_10_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void conv_11(
            float input[192][20][40],
            float weight[384][192],
            float bias[384],
            float output[384][20][40]
            )
{
    //cout << "conv_11..." << endl;

    for(int co = 0; co < 384; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 192; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                float result = sum + bias[co];
                output[co][h][w] = (result > 0)? result : 0.0f;
            }
        }
    }

    fo = fopen("conv_11_out", "w");
    for(int i = 0; i < 384; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "conv_11_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);
}


void conv_12(
            float input[384][20][40],
            float weight[10][384],
            float output[10][20][40]
            )
{
    //cout << "conv_12..." << endl;

    for(int co = 0; co < 10; co++) {
        for(int h = 0; h < 20; h++) {
            for(int w = 0; w < 40; w++) {
                float sum = 0;

                for(int ci = 0; ci < 384; ci++ ) {
                    sum += weight[co][ci] * input[ci][h][w];
                }
                output[co][h][w] = sum;
            }
        }
    }

    fo = fopen("conv_12_out", "w");
    for(int i = 0; i < 10; i++) {
        for(int j = 0; j < 20; j++) {
            for(int k = 0; k < 40; k ++) {
                fprintf(fo, "conv_12_output[%d][%d][%d] = %f\n", i, j, k, output[i][j][k]);
            }
        }
    }
    fclose(fo);

}




void compute_bounding_box( float input[10][20][40] )
{
	int batch_size = 1;
	int num_anchors = 2;
	int h = 20;
	int w = 40;

	float output[10][20][40];

    float box[4] = {1.4940052559648322, 2.3598481287086823, 4.0113013115312155, 5.760873975661669};

	float conf_thresh = 0.0;
	int conf_j = 0;
	int conf_m = 0;
	int conf_n = 0;

	//preprocessing anchor boxes xs and ys
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5][m][n] = 1/(1+exp(-input[j*5][m][n]))+n;
			}
		}
	}

	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
                output[j*5+1][m][n] = 1/(1+exp(-input[j*5+1][m][n]))+m;
			}
		}
	}
	//preprocessing anchor boxes ws and hs
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+2][m][n] = exp(input[j*5+2][m][n])*box[j*2];
			}
		}
	}
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+3][m][n] = exp(input[j*5+3][m][n])*box[j*2+1];
			}
		}
	}
	//preprocessing anchor boxes det_conf
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				output[j*5+4][m][n] = 1/(1+exp(-input[j*5+4][m][n]));
			}
		}
	}

	//find the maximum num
	for(int j = 0;j < num_anchors;j++){
		for(int m = 0;m < h;m++){
			for(int n = 0;n < w;n++){
				if(output[j*5+4][m][n] > conf_thresh){
					conf_thresh = output[j*5+4][m][n];
					conf_j = j;
					conf_m = m;
					conf_n = n;
				}
			}
		}
	}

	//calculate the output
	float predict_box[5] = {output[conf_j*5+0][conf_m][conf_n]/w,\
		output[conf_j*5+1][conf_m][conf_n]/h,\
		output[conf_j*5+2][conf_m][conf_n]/w,\
		output[conf_j*5+3][conf_m][conf_n]/h,\
		output[conf_j*5+4][conf_m][conf_n]};

	printf("Golden Model:\n");
    printf("conf_thresh: %f, conf_j: %d, conf_m: %d, conf_n: %d\n", conf_thresh, conf_j, conf_m, conf_n);
	for(int i = 0; i < 5; i++){
		printf("%f\n",predict_box[i]);
	}

	int x1, y1, x2, y2;

	x1 = (unsigned int)(((predict_box[0] - predict_box[2]/2.0) * 640));
	y1 = (unsigned int)(((predict_box[1] - predict_box[3]/2.0) * 360));
	x2 = (unsigned int)(((predict_box[0] + predict_box[2]/2.0) * 640));
	y2 = (unsigned int)(((predict_box[1] + predict_box[3]/2.0) * 360));

	printf("%d %d %d %d\n", x1, y1, x2, y2);
}






void golden_model()
{
	conv_1(image, conv_1_weight_tmp, conv_1_bias_tmp, conv_1_out);
	conv_2(conv_1_out, conv_2_weight_tmp, conv_2_bias_in, conv_2_out);
	max_pool_3(conv_2_out, pool_3_out);

	conv_4(pool_3_out, conv_4_weight_in, conv_4_bias_in, conv_4_out);
	conv_5(conv_4_out, conv_5_weight_in, conv_5_bias_in, conv_5_out);
	max_pool_6(conv_5_out, pool_6_out);

	conv_7(pool_6_out, conv_7_weight_in, conv_7_bias_in, conv_7_out);
	conv_8(conv_7_out, conv_8_weight_in, conv_8_bias_in, conv_8_out);
	max_pool_9(conv_8_out, pool_9_out);

	conv_10(pool_9_out, conv_10_weight_in, conv_10_bias_in, conv_10_out);
	conv_11(conv_10_out, conv_11_weight_in, conv_11_bias_in, conv_11_out);
	conv_12(conv_11_out, conv_12_weight_tmp, conv_12_out);

	compute_bounding_box(conv_12_out);


}


void fill_output( int layer, float buf[16][22][42], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 1; j <= 20; j++) {
			for(int k = 1; k <= 40; k++) {
				switch (layer)
				{
				case 1:
					conv_1_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 2:
					conv_2_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 4:
					conv_4_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 5:
					conv_5_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 7:
					conv_7_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 8:
					conv_8_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 10:
					conv_10_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 11:
					conv_11_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				case 12:
					conv_12_out_PL[ch*16+i][col*20+j-1][row*40+k-1] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}




void fill_output_pool( int layer, float buf[16][10][20], int ch, int col, int row)
{
	for(int i = 0; i < 16; i++) {
		for(int j = 0; j < 10; j++) {
			for(int k = 0; k < 20; k++) {
				switch (layer)
				{
				case 3:
					pool_3_out_PL[i + ch*16][j + col*10][k + row*20] = buf[i][j][k];
					break;
				case 6:
					pool_6_out_PL[i + ch*16][j + col*10][k + row*20] = buf[i][j][k];
					break;
				case 9:
					pool_9_out_PL[i + ch*16][j + col*10][k + row*20] = buf[i][j][k];
					break;
				default:
					printf("Wrong layer number.\n");
				}

			}
		}
	}
}


int PL_golden_compare_layer_1()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_1";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 3; ch++) {
			for(int w = 0; w < 320; w++) {
				for(int h = 0; h < 160; h++) {
				if( abs(conv_1_out_PL[ch][h][w] - conv_1_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_2()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_2";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
		for(int w = 0; w < 320; w++) {
			for(int h = 0; h < 160; h++) {

				if( abs(conv_2_out_PL[ch][h][w] - conv_2_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_3()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_3";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
			for(int w = 0; w < 160; w++) {
				for(int h = 0; h < 80; h++) {
				if( abs(pool_3_out_PL[ch][h][w] - pool_3_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}
	fclose(fo);


	return pass;
}


int PL_golden_compare_layer_4()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_4";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 48; ch++) {
			for(int w = 0; w < 160; w++) {
				for(int h = 0; h < 80; h++) {
				if( abs(conv_4_out_PL[ch][h][w] - conv_4_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_5()
{
	FILE* fo;
	int pass = 1;

	char* filename = "Comp_layer_5";
	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
			for(int w = 0; w < 160; w++) {
				for(int h = 0; h < 80; h++) {
				if( abs(conv_5_out_PL[ch][h][w] - conv_5_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}



int PL_golden_compare_layer_6()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_6");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
			for(int w = 0; w < 80; w++) {
				for(int h = 0; h < 40; h++) {
				if( abs(pool_6_out_PL[ch][h][w] - pool_6_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_7()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_7");


	fo = fopen(filename, "w");

	for(int ch = 0; ch < 96; ch++) {
			for(int w = 0; w < 80; w++) {
				for(int h = 0; h < 40; h++) {
				if( abs(conv_7_out_PL[ch][h][w] - conv_7_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_8()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_8");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
			for(int w = 0; w < 80; w++) {
				for(int h = 0; h < 40; h++) {
				if( abs(conv_8_out_PL[ch][h][w] - conv_8_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_9()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_9");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
			for(int w = 0; w < 40; w++) {
				for(int h = 0; h < 20; h++) {
					if( abs(pool_9_out_PL[ch][h][w] - pool_9_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_10()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_10");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 192; ch++) {
			for(int w = 0; w < 40; w++) {
				for(int h = 0; h < 20; h++) {
				if( abs(conv_10_out_PL[ch][h][w] - conv_10_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_11()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_11");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 384; ch++) {
			for(int w = 0; w < 40; w++) {
				for(int h = 0; h < 20; h++) {
				if( abs(conv_11_out_PL[ch][h][w] - conv_11_out[ch][h][w]) < EPSILON) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}


int PL_golden_compare_layer_12()
{
	FILE* fo;
	int pass = 1;

	char filename[32];
	sprintf(filename, "Comp_layer_12");

	fo = fopen(filename, "w");

	for(int ch = 0; ch < 10; ch++) {
			for(int w = 0; w < 40; w++) {
				for(int h = 0; h < 20; h++) {
				if( abs(conv_12_out_PL[ch][h][w] - conv_12_out[ch][h][w]) < EPSILON ) {
					fprintf(fo, ".");
				}
				else {
					fprintf(fo, "X");
				}
			}
			fprintf(fo, "\n");
		}
		fprintf(fo, "\n\n");
	}

	return pass;
}
