/**
  ******************************************************************************
  * @file    cnn.c
  * @author  AST Embedded Analytics Research Platform
  * @date    Fri Oct 25 11:36:35 2019
  * @brief   AI Tool Automatic Code Generator for Embedded NN computing
  ******************************************************************************
  * @attention
  *
  * Copyright (c) 2018 STMicroelectronics.
  * All rights reserved.
  *
  * This software component is licensed by ST under Ultimate Liberty license
  * SLA0044, the "License"; You may not use this file except in compliance with
  * the License. You may obtain a copy of the License at:
  *                             www.st.com/SLA0044
  *
  ******************************************************************************
  */



#include "cnn.h"

#include "ai_platform_interface.h"
#include "ai_math_helpers.h"

#include "core_common.h"
#include "layers.h"

#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#define AI_TOOLS_VERSION_MAJOR 4
#define AI_TOOLS_VERSION_MINOR 1
#define AI_TOOLS_VERSION_MICRO 0


#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#define AI_TOOLS_API_VERSION_MAJOR 1
#define AI_TOOLS_API_VERSION_MINOR 3
#define AI_TOOLS_API_VERSION_MICRO 0

#undef AI_NET_OBJ_INSTANCE
#define AI_NET_OBJ_INSTANCE g_cnn
 
#undef AI_CNN_MODEL_SIGNATURE
#define AI_CNN_MODEL_SIGNATURE     "b851251318365d0ef34d902ac1511f44"

#ifndef AI_TOOLS_REVISION_ID
#define AI_TOOLS_REVISION_ID     "(rev-4.1.0)"
#endif

#undef AI_TOOLS_DATE_TIME
#define AI_TOOLS_DATE_TIME   "Fri Oct 25 11:36:35 2019"

#undef AI_TOOLS_COMPILE_TIME
#define AI_TOOLS_COMPILE_TIME    __DATE__ " " __TIME__

#undef AI_CNN_N_BATCHES
#define AI_CNN_N_BATCHES         (1)

/**  Forward network declaration section  *************************************/
AI_STATIC ai_network AI_NET_OBJ_INSTANCE;


/**  Forward network array declarations  **************************************/
AI_STATIC ai_array conv2d_28_scratch0_array;   /* Array #0 */
AI_STATIC ai_array conv2d_26_scratch1_array;   /* Array #1 */
AI_STATIC ai_array conv2d_26_scratch0_array;   /* Array #2 */
AI_STATIC ai_array conv2d_25_scratch0_array;   /* Array #3 */
AI_STATIC ai_array conv2d_24_scratch0_array;   /* Array #4 */
AI_STATIC ai_array conv2d_23_scratch0_array;   /* Array #5 */
AI_STATIC ai_array conv2d_22_scratch0_array;   /* Array #6 */
AI_STATIC ai_array conv2d_21_scratch0_array;   /* Array #7 */
AI_STATIC ai_array conv2d_20_scratch0_array;   /* Array #8 */
AI_STATIC ai_array conv2d_19_scratch0_array;   /* Array #9 */
AI_STATIC ai_array conv2d_18_scratch0_array;   /* Array #10 */
AI_STATIC ai_array conv2d_17_scratch0_array;   /* Array #11 */
AI_STATIC ai_array conv2d_16_scratch0_array;   /* Array #12 */
AI_STATIC ai_array conv2d_15_scratch0_array;   /* Array #13 */
AI_STATIC ai_array conv2d_14_scratch0_array;   /* Array #14 */
AI_STATIC ai_array conv2d_13_scratch0_array;   /* Array #15 */
AI_STATIC ai_array conv2d_12_scratch0_array;   /* Array #16 */
AI_STATIC ai_array conv2d_11_scratch0_array;   /* Array #17 */
AI_STATIC ai_array conv2d_10_scratch0_array;   /* Array #18 */
AI_STATIC ai_array conv2d_9_scratch0_array;   /* Array #19 */
AI_STATIC ai_array conv2d_8_scratch0_array;   /* Array #20 */
AI_STATIC ai_array conv2d_7_scratch0_array;   /* Array #21 */
AI_STATIC ai_array conv2d_6_scratch0_array;   /* Array #22 */
AI_STATIC ai_array conv2d_5_scratch0_array;   /* Array #23 */
AI_STATIC ai_array conv2d_4_scratch0_array;   /* Array #24 */
AI_STATIC ai_array conv2d_3_scratch0_array;   /* Array #25 */
AI_STATIC ai_array conv2d_2_scratch0_array;   /* Array #26 */
AI_STATIC ai_array conv2d_1_scratch0_array;   /* Array #27 */
AI_STATIC ai_array conv2d_0_scratch0_array;   /* Array #28 */
AI_STATIC ai_array conv2d_28_bias_array;   /* Array #29 */
AI_STATIC ai_array conv2d_28_weights_array;   /* Array #30 */
AI_STATIC ai_array conv2d_26_bias_array;   /* Array #31 */
AI_STATIC ai_array conv2d_26_weights_array;   /* Array #32 */
AI_STATIC ai_array conv2d_25_bias_array;   /* Array #33 */
AI_STATIC ai_array conv2d_25_weights_array;   /* Array #34 */
AI_STATIC ai_array conv2d_24_bias_array;   /* Array #35 */
AI_STATIC ai_array conv2d_24_weights_array;   /* Array #36 */
AI_STATIC ai_array conv2d_23_bias_array;   /* Array #37 */
AI_STATIC ai_array conv2d_23_weights_array;   /* Array #38 */
AI_STATIC ai_array conv2d_22_bias_array;   /* Array #39 */
AI_STATIC ai_array conv2d_22_weights_array;   /* Array #40 */
AI_STATIC ai_array conv2d_21_bias_array;   /* Array #41 */
AI_STATIC ai_array conv2d_21_weights_array;   /* Array #42 */
AI_STATIC ai_array conv2d_20_bias_array;   /* Array #43 */
AI_STATIC ai_array conv2d_20_weights_array;   /* Array #44 */
AI_STATIC ai_array conv2d_19_bias_array;   /* Array #45 */
AI_STATIC ai_array conv2d_19_weights_array;   /* Array #46 */
AI_STATIC ai_array conv2d_18_bias_array;   /* Array #47 */
AI_STATIC ai_array conv2d_18_weights_array;   /* Array #48 */
AI_STATIC ai_array conv2d_17_bias_array;   /* Array #49 */
AI_STATIC ai_array conv2d_17_weights_array;   /* Array #50 */
AI_STATIC ai_array conv2d_16_bias_array;   /* Array #51 */
AI_STATIC ai_array conv2d_16_weights_array;   /* Array #52 */
AI_STATIC ai_array conv2d_15_bias_array;   /* Array #53 */
AI_STATIC ai_array conv2d_15_weights_array;   /* Array #54 */
AI_STATIC ai_array conv2d_14_bias_array;   /* Array #55 */
AI_STATIC ai_array conv2d_14_weights_array;   /* Array #56 */
AI_STATIC ai_array conv2d_13_bias_array;   /* Array #57 */
AI_STATIC ai_array conv2d_13_weights_array;   /* Array #58 */
AI_STATIC ai_array conv2d_12_bias_array;   /* Array #59 */
AI_STATIC ai_array conv2d_12_weights_array;   /* Array #60 */
AI_STATIC ai_array conv2d_11_bias_array;   /* Array #61 */
AI_STATIC ai_array conv2d_11_weights_array;   /* Array #62 */
AI_STATIC ai_array conv2d_10_bias_array;   /* Array #63 */
AI_STATIC ai_array conv2d_10_weights_array;   /* Array #64 */
AI_STATIC ai_array conv2d_9_bias_array;   /* Array #65 */
AI_STATIC ai_array conv2d_9_weights_array;   /* Array #66 */
AI_STATIC ai_array conv2d_8_bias_array;   /* Array #67 */
AI_STATIC ai_array conv2d_8_weights_array;   /* Array #68 */
AI_STATIC ai_array conv2d_7_bias_array;   /* Array #69 */
AI_STATIC ai_array conv2d_7_weights_array;   /* Array #70 */
AI_STATIC ai_array conv2d_6_bias_array;   /* Array #71 */
AI_STATIC ai_array conv2d_6_weights_array;   /* Array #72 */
AI_STATIC ai_array conv2d_5_bias_array;   /* Array #73 */
AI_STATIC ai_array conv2d_5_weights_array;   /* Array #74 */
AI_STATIC ai_array conv2d_4_bias_array;   /* Array #75 */
AI_STATIC ai_array conv2d_4_weights_array;   /* Array #76 */
AI_STATIC ai_array conv2d_3_bias_array;   /* Array #77 */
AI_STATIC ai_array conv2d_3_weights_array;   /* Array #78 */
AI_STATIC ai_array conv2d_2_bias_array;   /* Array #79 */
AI_STATIC ai_array conv2d_2_weights_array;   /* Array #80 */
AI_STATIC ai_array conv2d_1_bias_array;   /* Array #81 */
AI_STATIC ai_array conv2d_1_weights_array;   /* Array #82 */
AI_STATIC ai_array conv2d_0_bias_array;   /* Array #83 */
AI_STATIC ai_array conv2d_0_weights_array;   /* Array #84 */
AI_STATIC ai_array input_0_output_array;   /* Array #85 */
AI_STATIC ai_array conv2d_0_output_array;   /* Array #86 */
AI_STATIC ai_array conv2d_1_output_array;   /* Array #87 */
AI_STATIC ai_array conv2d_2_output_array;   /* Array #88 */
AI_STATIC ai_array conv2d_3_output_array;   /* Array #89 */
AI_STATIC ai_array conv2d_4_output_array;   /* Array #90 */
AI_STATIC ai_array conv2d_5_output_array;   /* Array #91 */
AI_STATIC ai_array conv2d_6_output_array;   /* Array #92 */
AI_STATIC ai_array conv2d_7_output_array;   /* Array #93 */
AI_STATIC ai_array conv2d_8_output_array;   /* Array #94 */
AI_STATIC ai_array conv2d_9_output_array;   /* Array #95 */
AI_STATIC ai_array conv2d_10_output_array;   /* Array #96 */
AI_STATIC ai_array conv2d_11_output_array;   /* Array #97 */
AI_STATIC ai_array conv2d_12_output_array;   /* Array #98 */
AI_STATIC ai_array conv2d_13_output_array;   /* Array #99 */
AI_STATIC ai_array conv2d_14_output_array;   /* Array #100 */
AI_STATIC ai_array conv2d_15_output_array;   /* Array #101 */
AI_STATIC ai_array conv2d_16_output_array;   /* Array #102 */
AI_STATIC ai_array conv2d_17_output_array;   /* Array #103 */
AI_STATIC ai_array conv2d_18_output_array;   /* Array #104 */
AI_STATIC ai_array conv2d_19_output_array;   /* Array #105 */
AI_STATIC ai_array conv2d_20_output_array;   /* Array #106 */
AI_STATIC ai_array conv2d_21_output_array;   /* Array #107 */
AI_STATIC ai_array conv2d_22_output_array;   /* Array #108 */
AI_STATIC ai_array conv2d_23_output_array;   /* Array #109 */
AI_STATIC ai_array conv2d_24_output_array;   /* Array #110 */
AI_STATIC ai_array conv2d_25_output_array;   /* Array #111 */
AI_STATIC ai_array conv2d_26_output_array;   /* Array #112 */
AI_STATIC ai_array conv2d_28_output_array;   /* Array #113 */
AI_STATIC ai_array reshape_29_fmt_output_array;   /* Array #114 */
AI_STATIC ai_array nl_30_output_array;   /* Array #115 */
AI_STATIC ai_array nl_30_fmt_output_array;   /* Array #116 */


/**  Forward network tensor declarations  *************************************/
AI_STATIC ai_tensor conv2d_28_scratch0;   /* Tensor #0 */
AI_STATIC ai_tensor conv2d_26_scratch1;   /* Tensor #1 */
AI_STATIC ai_tensor conv2d_26_scratch0;   /* Tensor #2 */
AI_STATIC ai_tensor conv2d_25_scratch0;   /* Tensor #3 */
AI_STATIC ai_tensor conv2d_24_scratch0;   /* Tensor #4 */
AI_STATIC ai_tensor conv2d_23_scratch0;   /* Tensor #5 */
AI_STATIC ai_tensor conv2d_22_scratch0;   /* Tensor #6 */
AI_STATIC ai_tensor conv2d_21_scratch0;   /* Tensor #7 */
AI_STATIC ai_tensor conv2d_20_scratch0;   /* Tensor #8 */
AI_STATIC ai_tensor conv2d_19_scratch0;   /* Tensor #9 */
AI_STATIC ai_tensor conv2d_18_scratch0;   /* Tensor #10 */
AI_STATIC ai_tensor conv2d_17_scratch0;   /* Tensor #11 */
AI_STATIC ai_tensor conv2d_16_scratch0;   /* Tensor #12 */
AI_STATIC ai_tensor conv2d_15_scratch0;   /* Tensor #13 */
AI_STATIC ai_tensor conv2d_14_scratch0;   /* Tensor #14 */
AI_STATIC ai_tensor conv2d_13_scratch0;   /* Tensor #15 */
AI_STATIC ai_tensor conv2d_12_scratch0;   /* Tensor #16 */
AI_STATIC ai_tensor conv2d_11_scratch0;   /* Tensor #17 */
AI_STATIC ai_tensor conv2d_10_scratch0;   /* Tensor #18 */
AI_STATIC ai_tensor conv2d_9_scratch0;   /* Tensor #19 */
AI_STATIC ai_tensor conv2d_8_scratch0;   /* Tensor #20 */
AI_STATIC ai_tensor conv2d_7_scratch0;   /* Tensor #21 */
AI_STATIC ai_tensor conv2d_6_scratch0;   /* Tensor #22 */
AI_STATIC ai_tensor conv2d_5_scratch0;   /* Tensor #23 */
AI_STATIC ai_tensor conv2d_4_scratch0;   /* Tensor #24 */
AI_STATIC ai_tensor conv2d_3_scratch0;   /* Tensor #25 */
AI_STATIC ai_tensor conv2d_2_scratch0;   /* Tensor #26 */
AI_STATIC ai_tensor conv2d_1_scratch0;   /* Tensor #27 */
AI_STATIC ai_tensor conv2d_0_scratch0;   /* Tensor #28 */
AI_STATIC ai_tensor conv2d_28_bias;   /* Tensor #29 */
AI_STATIC ai_tensor conv2d_28_weights;   /* Tensor #30 */
AI_STATIC ai_tensor conv2d_26_bias;   /* Tensor #31 */
AI_STATIC ai_tensor conv2d_26_weights;   /* Tensor #32 */
AI_STATIC ai_tensor conv2d_25_bias;   /* Tensor #33 */
AI_STATIC ai_tensor conv2d_25_weights;   /* Tensor #34 */
AI_STATIC ai_tensor conv2d_24_bias;   /* Tensor #35 */
AI_STATIC ai_tensor conv2d_24_weights;   /* Tensor #36 */
AI_STATIC ai_tensor conv2d_23_bias;   /* Tensor #37 */
AI_STATIC ai_tensor conv2d_23_weights;   /* Tensor #38 */
AI_STATIC ai_tensor conv2d_22_bias;   /* Tensor #39 */
AI_STATIC ai_tensor conv2d_22_weights;   /* Tensor #40 */
AI_STATIC ai_tensor conv2d_21_bias;   /* Tensor #41 */
AI_STATIC ai_tensor conv2d_21_weights;   /* Tensor #42 */
AI_STATIC ai_tensor conv2d_20_bias;   /* Tensor #43 */
AI_STATIC ai_tensor conv2d_20_weights;   /* Tensor #44 */
AI_STATIC ai_tensor conv2d_19_bias;   /* Tensor #45 */
AI_STATIC ai_tensor conv2d_19_weights;   /* Tensor #46 */
AI_STATIC ai_tensor conv2d_18_bias;   /* Tensor #47 */
AI_STATIC ai_tensor conv2d_18_weights;   /* Tensor #48 */
AI_STATIC ai_tensor conv2d_17_bias;   /* Tensor #49 */
AI_STATIC ai_tensor conv2d_17_weights;   /* Tensor #50 */
AI_STATIC ai_tensor conv2d_16_bias;   /* Tensor #51 */
AI_STATIC ai_tensor conv2d_16_weights;   /* Tensor #52 */
AI_STATIC ai_tensor conv2d_15_bias;   /* Tensor #53 */
AI_STATIC ai_tensor conv2d_15_weights;   /* Tensor #54 */
AI_STATIC ai_tensor conv2d_14_bias;   /* Tensor #55 */
AI_STATIC ai_tensor conv2d_14_weights;   /* Tensor #56 */
AI_STATIC ai_tensor conv2d_13_bias;   /* Tensor #57 */
AI_STATIC ai_tensor conv2d_13_weights;   /* Tensor #58 */
AI_STATIC ai_tensor conv2d_12_bias;   /* Tensor #59 */
AI_STATIC ai_tensor conv2d_12_weights;   /* Tensor #60 */
AI_STATIC ai_tensor conv2d_11_bias;   /* Tensor #61 */
AI_STATIC ai_tensor conv2d_11_weights;   /* Tensor #62 */
AI_STATIC ai_tensor conv2d_10_bias;   /* Tensor #63 */
AI_STATIC ai_tensor conv2d_10_weights;   /* Tensor #64 */
AI_STATIC ai_tensor conv2d_9_bias;   /* Tensor #65 */
AI_STATIC ai_tensor conv2d_9_weights;   /* Tensor #66 */
AI_STATIC ai_tensor conv2d_8_bias;   /* Tensor #67 */
AI_STATIC ai_tensor conv2d_8_weights;   /* Tensor #68 */
AI_STATIC ai_tensor conv2d_7_bias;   /* Tensor #69 */
AI_STATIC ai_tensor conv2d_7_weights;   /* Tensor #70 */
AI_STATIC ai_tensor conv2d_6_bias;   /* Tensor #71 */
AI_STATIC ai_tensor conv2d_6_weights;   /* Tensor #72 */
AI_STATIC ai_tensor conv2d_5_bias;   /* Tensor #73 */
AI_STATIC ai_tensor conv2d_5_weights;   /* Tensor #74 */
AI_STATIC ai_tensor conv2d_4_bias;   /* Tensor #75 */
AI_STATIC ai_tensor conv2d_4_weights;   /* Tensor #76 */
AI_STATIC ai_tensor conv2d_3_bias;   /* Tensor #77 */
AI_STATIC ai_tensor conv2d_3_weights;   /* Tensor #78 */
AI_STATIC ai_tensor conv2d_2_bias;   /* Tensor #79 */
AI_STATIC ai_tensor conv2d_2_weights;   /* Tensor #80 */
AI_STATIC ai_tensor conv2d_1_bias;   /* Tensor #81 */
AI_STATIC ai_tensor conv2d_1_weights;   /* Tensor #82 */
AI_STATIC ai_tensor conv2d_0_bias;   /* Tensor #83 */
AI_STATIC ai_tensor conv2d_0_weights;   /* Tensor #84 */
AI_STATIC ai_tensor input_0_output;   /* Tensor #85 */
AI_STATIC ai_tensor conv2d_0_output;   /* Tensor #86 */
AI_STATIC ai_tensor conv2d_1_output;   /* Tensor #87 */
AI_STATIC ai_tensor conv2d_2_output;   /* Tensor #88 */
AI_STATIC ai_tensor conv2d_3_output;   /* Tensor #89 */
AI_STATIC ai_tensor conv2d_4_output;   /* Tensor #90 */
AI_STATIC ai_tensor conv2d_5_output;   /* Tensor #91 */
AI_STATIC ai_tensor conv2d_6_output;   /* Tensor #92 */
AI_STATIC ai_tensor conv2d_7_output;   /* Tensor #93 */
AI_STATIC ai_tensor conv2d_8_output;   /* Tensor #94 */
AI_STATIC ai_tensor conv2d_9_output;   /* Tensor #95 */
AI_STATIC ai_tensor conv2d_10_output;   /* Tensor #96 */
AI_STATIC ai_tensor conv2d_11_output;   /* Tensor #97 */
AI_STATIC ai_tensor conv2d_12_output;   /* Tensor #98 */
AI_STATIC ai_tensor conv2d_13_output;   /* Tensor #99 */
AI_STATIC ai_tensor conv2d_14_output;   /* Tensor #100 */
AI_STATIC ai_tensor conv2d_15_output;   /* Tensor #101 */
AI_STATIC ai_tensor conv2d_16_output;   /* Tensor #102 */
AI_STATIC ai_tensor conv2d_17_output;   /* Tensor #103 */
AI_STATIC ai_tensor conv2d_18_output;   /* Tensor #104 */
AI_STATIC ai_tensor conv2d_19_output;   /* Tensor #105 */
AI_STATIC ai_tensor conv2d_20_output;   /* Tensor #106 */
AI_STATIC ai_tensor conv2d_21_output;   /* Tensor #107 */
AI_STATIC ai_tensor conv2d_22_output;   /* Tensor #108 */
AI_STATIC ai_tensor conv2d_23_output;   /* Tensor #109 */
AI_STATIC ai_tensor conv2d_24_output;   /* Tensor #110 */
AI_STATIC ai_tensor conv2d_25_output;   /* Tensor #111 */
AI_STATIC ai_tensor conv2d_26_output;   /* Tensor #112 */
AI_STATIC ai_tensor conv2d_28_output;   /* Tensor #113 */
AI_STATIC ai_tensor reshape_29_fmt_output;   /* Tensor #114 */
AI_STATIC ai_tensor nl_30_output;   /* Tensor #115 */
AI_STATIC ai_tensor nl_30_fmt_output;   /* Tensor #116 */


/**  Forward network tensor chain declarations  *******************************/
AI_STATIC_CONST ai_tensor_chain conv2d_0_chain;   /* Chain #0 */
AI_STATIC_CONST ai_tensor_chain conv2d_1_chain;   /* Chain #1 */
AI_STATIC_CONST ai_tensor_chain conv2d_2_chain;   /* Chain #2 */
AI_STATIC_CONST ai_tensor_chain conv2d_3_chain;   /* Chain #3 */
AI_STATIC_CONST ai_tensor_chain conv2d_4_chain;   /* Chain #4 */
AI_STATIC_CONST ai_tensor_chain conv2d_5_chain;   /* Chain #5 */
AI_STATIC_CONST ai_tensor_chain conv2d_6_chain;   /* Chain #6 */
AI_STATIC_CONST ai_tensor_chain conv2d_7_chain;   /* Chain #7 */
AI_STATIC_CONST ai_tensor_chain conv2d_8_chain;   /* Chain #8 */
AI_STATIC_CONST ai_tensor_chain conv2d_9_chain;   /* Chain #9 */
AI_STATIC_CONST ai_tensor_chain conv2d_10_chain;   /* Chain #10 */
AI_STATIC_CONST ai_tensor_chain conv2d_11_chain;   /* Chain #11 */
AI_STATIC_CONST ai_tensor_chain conv2d_12_chain;   /* Chain #12 */
AI_STATIC_CONST ai_tensor_chain conv2d_13_chain;   /* Chain #13 */
AI_STATIC_CONST ai_tensor_chain conv2d_14_chain;   /* Chain #14 */
AI_STATIC_CONST ai_tensor_chain conv2d_15_chain;   /* Chain #15 */
AI_STATIC_CONST ai_tensor_chain conv2d_16_chain;   /* Chain #16 */
AI_STATIC_CONST ai_tensor_chain conv2d_17_chain;   /* Chain #17 */
AI_STATIC_CONST ai_tensor_chain conv2d_18_chain;   /* Chain #18 */
AI_STATIC_CONST ai_tensor_chain conv2d_19_chain;   /* Chain #19 */
AI_STATIC_CONST ai_tensor_chain conv2d_20_chain;   /* Chain #20 */
AI_STATIC_CONST ai_tensor_chain conv2d_21_chain;   /* Chain #21 */
AI_STATIC_CONST ai_tensor_chain conv2d_22_chain;   /* Chain #22 */
AI_STATIC_CONST ai_tensor_chain conv2d_23_chain;   /* Chain #23 */
AI_STATIC_CONST ai_tensor_chain conv2d_24_chain;   /* Chain #24 */
AI_STATIC_CONST ai_tensor_chain conv2d_25_chain;   /* Chain #25 */
AI_STATIC_CONST ai_tensor_chain conv2d_26_chain;   /* Chain #26 */
AI_STATIC_CONST ai_tensor_chain conv2d_28_chain;   /* Chain #27 */
AI_STATIC_CONST ai_tensor_chain reshape_29_fmt_chain;   /* Chain #28 */
AI_STATIC_CONST ai_tensor_chain nl_30_chain;   /* Chain #29 */
AI_STATIC_CONST ai_tensor_chain nl_30_fmt_chain;   /* Chain #30 */


/**  Subgraph network operator tensor chain declarations  *********************/


/**  Subgraph network operator declarations  *********************************/


/**  Forward network layer declarations  **************************************/
AI_STATIC ai_layer_conv2d conv2d_0_layer; /* Layer #0 */
AI_STATIC ai_layer_conv2d conv2d_1_layer; /* Layer #1 */
AI_STATIC ai_layer_conv2d conv2d_2_layer; /* Layer #2 */
AI_STATIC ai_layer_conv2d conv2d_3_layer; /* Layer #3 */
AI_STATIC ai_layer_conv2d conv2d_4_layer; /* Layer #4 */
AI_STATIC ai_layer_conv2d conv2d_5_layer; /* Layer #5 */
AI_STATIC ai_layer_conv2d conv2d_6_layer; /* Layer #6 */
AI_STATIC ai_layer_conv2d conv2d_7_layer; /* Layer #7 */
AI_STATIC ai_layer_conv2d conv2d_8_layer; /* Layer #8 */
AI_STATIC ai_layer_conv2d conv2d_9_layer; /* Layer #9 */
AI_STATIC ai_layer_conv2d conv2d_10_layer; /* Layer #10 */
AI_STATIC ai_layer_conv2d conv2d_11_layer; /* Layer #11 */
AI_STATIC ai_layer_conv2d conv2d_12_layer; /* Layer #12 */
AI_STATIC ai_layer_conv2d conv2d_13_layer; /* Layer #13 */
AI_STATIC ai_layer_conv2d conv2d_14_layer; /* Layer #14 */
AI_STATIC ai_layer_conv2d conv2d_15_layer; /* Layer #15 */
AI_STATIC ai_layer_conv2d conv2d_16_layer; /* Layer #16 */
AI_STATIC ai_layer_conv2d conv2d_17_layer; /* Layer #17 */
AI_STATIC ai_layer_conv2d conv2d_18_layer; /* Layer #18 */
AI_STATIC ai_layer_conv2d conv2d_19_layer; /* Layer #19 */
AI_STATIC ai_layer_conv2d conv2d_20_layer; /* Layer #20 */
AI_STATIC ai_layer_conv2d conv2d_21_layer; /* Layer #21 */
AI_STATIC ai_layer_conv2d conv2d_22_layer; /* Layer #22 */
AI_STATIC ai_layer_conv2d conv2d_23_layer; /* Layer #23 */
AI_STATIC ai_layer_conv2d conv2d_24_layer; /* Layer #24 */
AI_STATIC ai_layer_conv2d conv2d_25_layer; /* Layer #25 */
AI_STATIC ai_layer_conv2d_nl_pool conv2d_26_layer; /* Layer #26 */
AI_STATIC ai_layer_conv2d conv2d_28_layer; /* Layer #27 */
AI_STATIC ai_layer_nl reshape_29_fmt_layer; /* Layer #28 */
AI_STATIC ai_layer_nl nl_30_layer; /* Layer #29 */
AI_STATIC ai_layer_nl nl_30_fmt_layer; /* Layer #30 */


/**  Array declarations section  **********************************************/
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch1_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 6400,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1024,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 6913,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 3457,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1729,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 1729,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 865,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 865,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 433,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 32,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 217,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_scratch0_array, AI_ARRAY_FORMAT_S8,
  NULL, NULL, 140,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 1001,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 256256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 65536,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 2304,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 32768,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 16384,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 16384,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 16384,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 16384,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 16384,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1152,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 8192,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 576,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 4096,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 576,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 64,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 2048,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 288,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1024,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 288,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 32,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 512,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 144,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 16,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 128,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 72,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_bias_array, AI_ARRAY_FORMAT_S32,
  NULL, NULL, 8,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_weights_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 216,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  input_0_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 76800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_0_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 51200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_1_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 51200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_2_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 102400,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_3_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 25600,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_4_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 51200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_5_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 51200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_6_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 51200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_7_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_8_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 25600,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_9_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 25600,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_10_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 25600,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_11_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 6400,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_12_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_13_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_14_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_15_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_16_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_17_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_18_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_19_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_20_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_21_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_22_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 12800,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_23_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 3200,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_24_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 6400,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_25_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 6400,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_26_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 256,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  conv2d_28_output_array, AI_ARRAY_FORMAT_U8,
  NULL, NULL, 1001,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  reshape_29_fmt_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1001,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  nl_30_output_array, AI_ARRAY_FORMAT_FLOAT,
  NULL, NULL, 1001,
  AI_STATIC)
AI_ARRAY_OBJ_DECLARE(
  nl_30_fmt_output_array, AI_ARRAY_FORMAT_U8|AI_FMT_FLAG_IS_IO,
  NULL, NULL, 1001,
  AI_STATIC)


AI_STATIC ai_intq_info_list conv2d_26_scratch1_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #0 */
AI_STATIC ai_intq_info_list conv2d_28_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00014470858150161803f), AI_PACK_INTQ_ZP(0)));   /* Int quant #1 */
AI_STATIC ai_intq_info_list conv2d_28_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006150359287858009f), AI_PACK_UINTQ_ZP(85)));   /* Int quant #2 */
AI_STATIC ai_intq_info_list conv2d_26_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007497151964344084f), AI_PACK_INTQ_ZP(0)));   /* Int quant #3 */
AI_STATIC ai_intq_info_list conv2d_26_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03186416253447533f), AI_PACK_UINTQ_ZP(154)));   /* Int quant #4 */
AI_STATIC ai_intq_info_list conv2d_25_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007364358170889318f), AI_PACK_INTQ_ZP(0)));   /* Int quant #5 */
AI_STATIC ai_intq_info_list conv2d_25_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03129976615309715f), AI_PACK_UINTQ_ZP(178)));   /* Int quant #6 */
AI_STATIC ai_intq_info_list conv2d_24_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002700200129766017f), AI_PACK_INTQ_ZP(0)));   /* Int quant #7 */
AI_STATIC ai_intq_info_list conv2d_24_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.01147630624473095f), AI_PACK_UINTQ_ZP(87)));   /* Int quant #8 */
AI_STATIC ai_intq_info_list conv2d_23_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007352353422902524f), AI_PACK_INTQ_ZP(0)));   /* Int quant #9 */
AI_STATIC ai_intq_info_list conv2d_23_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03124874457716942f), AI_PACK_UINTQ_ZP(65)));   /* Int quant #10 */
AI_STATIC ai_intq_info_list conv2d_22_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00017431839660275728f), AI_PACK_INTQ_ZP(0)));   /* Int quant #11 */
AI_STATIC ai_intq_info_list conv2d_22_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.007408826146274805f), AI_PACK_UINTQ_ZP(124)));   /* Int quant #12 */
AI_STATIC ai_intq_info_list conv2d_21_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000567332492209971f), AI_PACK_INTQ_ZP(0)));   /* Int quant #13 */
AI_STATIC ai_intq_info_list conv2d_21_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.02411258965730667f), AI_PACK_UINTQ_ZP(181)));   /* Int quant #14 */
AI_STATIC ai_intq_info_list conv2d_20_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000308339687762782f), AI_PACK_INTQ_ZP(0)));   /* Int quant #15 */
AI_STATIC ai_intq_info_list conv2d_20_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013104956597089767f), AI_PACK_UINTQ_ZP(112)));   /* Int quant #16 */
AI_STATIC ai_intq_info_list conv2d_19_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0004303431778680533f), AI_PACK_INTQ_ZP(0)));   /* Int quant #17 */
AI_STATIC ai_intq_info_list conv2d_19_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018290311098098755f), AI_PACK_UINTQ_ZP(136)));   /* Int quant #18 */
AI_STATIC ai_intq_info_list conv2d_18_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0002532463113311678f), AI_PACK_INTQ_ZP(0)));   /* Int quant #19 */
AI_STATIC ai_intq_info_list conv2d_18_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010763395577669144f), AI_PACK_UINTQ_ZP(113)));   /* Int quant #20 */
AI_STATIC ai_intq_info_list conv2d_17_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000466475379653275f), AI_PACK_INTQ_ZP(0)));   /* Int quant #21 */
AI_STATIC ai_intq_info_list conv2d_17_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.019825991243124008f), AI_PACK_UINTQ_ZP(108)));   /* Int quant #22 */
AI_STATIC ai_intq_info_list conv2d_16_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00029948094743303955f), AI_PACK_INTQ_ZP(0)));   /* Int quant #23 */
AI_STATIC ai_intq_info_list conv2d_16_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.012728446163237095f), AI_PACK_UINTQ_ZP(169)));   /* Int quant #24 */
AI_STATIC ai_intq_info_list conv2d_15_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0008384188986383379f), AI_PACK_INTQ_ZP(0)));   /* Int quant #25 */
AI_STATIC ai_intq_info_list conv2d_15_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.03563421964645386f), AI_PACK_UINTQ_ZP(161)));   /* Int quant #26 */
AI_STATIC ai_intq_info_list conv2d_14_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00020510370086412877f), AI_PACK_INTQ_ZP(0)));   /* Int quant #27 */
AI_STATIC ai_intq_info_list conv2d_14_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.008717253804206848f), AI_PACK_UINTQ_ZP(136)));   /* Int quant #28 */
AI_STATIC ai_intq_info_list conv2d_13_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0011808548588305712f), AI_PACK_INTQ_ZP(0)));   /* Int quant #29 */
AI_STATIC ai_intq_info_list conv2d_13_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0501883290708065f), AI_PACK_UINTQ_ZP(152)));   /* Int quant #30 */
AI_STATIC ai_intq_info_list conv2d_12_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00032609282061457634f), AI_PACK_INTQ_ZP(0)));   /* Int quant #31 */
AI_STATIC ai_intq_info_list conv2d_12_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013859495520591736f), AI_PACK_UINTQ_ZP(143)));   /* Int quant #32 */
AI_STATIC ai_intq_info_list conv2d_11_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00021771196043118834f), AI_PACK_INTQ_ZP(0)));   /* Int quant #33 */
AI_STATIC ai_intq_info_list conv2d_11_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009253126569092274f), AI_PACK_UINTQ_ZP(150)));   /* Int quant #34 */
AI_STATIC ai_intq_info_list conv2d_10_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.000441147421952337f), AI_PACK_INTQ_ZP(0)));   /* Int quant #35 */
AI_STATIC ai_intq_info_list conv2d_10_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018749510869383812f), AI_PACK_UINTQ_ZP(112)));   /* Int quant #36 */
AI_STATIC ai_intq_info_list conv2d_9_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0007299647550098598f), AI_PACK_INTQ_ZP(0)));   /* Int quant #37 */
AI_STATIC ai_intq_info_list conv2d_9_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.031024735420942307f), AI_PACK_UINTQ_ZP(133)));   /* Int quant #38 */
AI_STATIC ai_intq_info_list conv2d_8_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00044083379907533526f), AI_PACK_INTQ_ZP(0)));   /* Int quant #39 */
AI_STATIC ai_intq_info_list conv2d_8_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.018736179918050766f), AI_PACK_UINTQ_ZP(133)));   /* Int quant #40 */
AI_STATIC ai_intq_info_list conv2d_7_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00031582434894517064f), AI_PACK_INTQ_ZP(0)));   /* Int quant #41 */
AI_STATIC ai_intq_info_list conv2d_7_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.013423068448901176f), AI_PACK_UINTQ_ZP(130)));   /* Int quant #42 */
AI_STATIC ai_intq_info_list conv2d_6_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005556192481890321f), AI_PACK_INTQ_ZP(0)));   /* Int quant #43 */
AI_STATIC ai_intq_info_list conv2d_6_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023614754900336266f), AI_PACK_UINTQ_ZP(97)));   /* Int quant #44 */
AI_STATIC ai_intq_info_list conv2d_5_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0023195857647806406f), AI_PACK_INTQ_ZP(0)));   /* Int quant #45 */
AI_STATIC ai_intq_info_list conv2d_5_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0985863134264946f), AI_PACK_UINTQ_ZP(125)));   /* Int quant #46 */
AI_STATIC ai_intq_info_list conv2d_4_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0005890397005714476f), AI_PACK_INTQ_ZP(0)));   /* Int quant #47 */
AI_STATIC ai_intq_info_list conv2d_4_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.025035183876752853f), AI_PACK_UINTQ_ZP(105)));   /* Int quant #48 */
AI_STATIC ai_intq_info_list conv2d_3_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0035230836365371943f), AI_PACK_INTQ_ZP(0)));   /* Int quant #49 */
AI_STATIC ai_intq_info_list conv2d_3_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.14973700046539307f), AI_PACK_UINTQ_ZP(67)));   /* Int quant #50 */
AI_STATIC ai_intq_info_list conv2d_2_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00025867955991998315f), AI_PACK_INTQ_ZP(0)));   /* Int quant #51 */
AI_STATIC ai_intq_info_list conv2d_2_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.010994317941367626f), AI_PACK_UINTQ_ZP(179)));   /* Int quant #52 */
AI_STATIC ai_intq_info_list conv2d_1_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.009114247746765614f), AI_PACK_INTQ_ZP(0)));   /* Int quant #53 */
AI_STATIC ai_intq_info_list conv2d_1_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.38737091422080994f), AI_PACK_UINTQ_ZP(75)));   /* Int quant #54 */
AI_STATIC ai_intq_info_list conv2d_0_bias_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_S8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(5.383966345107183e-05f), AI_PACK_INTQ_ZP(0)));   /* Int quant #55 */
AI_STATIC ai_intq_info_list conv2d_0_weights_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.006891476921737194f), AI_PACK_UINTQ_ZP(100)));   /* Int quant #56 */
AI_STATIC ai_intq_info_list input_0_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.0078125f), AI_PACK_UINTQ_ZP(128)));   /* Int quant #57 */
AI_STATIC ai_intq_info_list conv2d_0_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #58 */
AI_STATIC ai_intq_info_list conv2d_1_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #59 */
AI_STATIC ai_intq_info_list conv2d_2_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #60 */
AI_STATIC ai_intq_info_list conv2d_3_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #61 */
AI_STATIC ai_intq_info_list conv2d_4_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #62 */
AI_STATIC ai_intq_info_list conv2d_5_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #63 */
AI_STATIC ai_intq_info_list conv2d_6_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #64 */
AI_STATIC ai_intq_info_list conv2d_7_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #65 */
AI_STATIC ai_intq_info_list conv2d_8_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #66 */
AI_STATIC ai_intq_info_list conv2d_9_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #67 */
AI_STATIC ai_intq_info_list conv2d_10_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #68 */
AI_STATIC ai_intq_info_list conv2d_11_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #69 */
AI_STATIC ai_intq_info_list conv2d_12_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #70 */
AI_STATIC ai_intq_info_list conv2d_13_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #71 */
AI_STATIC ai_intq_info_list conv2d_14_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #72 */
AI_STATIC ai_intq_info_list conv2d_15_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #73 */
AI_STATIC ai_intq_info_list conv2d_16_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #74 */
AI_STATIC ai_intq_info_list conv2d_17_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #75 */
AI_STATIC ai_intq_info_list conv2d_18_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #76 */
AI_STATIC ai_intq_info_list conv2d_19_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #77 */
AI_STATIC ai_intq_info_list conv2d_20_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #78 */
AI_STATIC ai_intq_info_list conv2d_21_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #79 */
AI_STATIC ai_intq_info_list conv2d_22_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #80 */
AI_STATIC ai_intq_info_list conv2d_23_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #81 */
AI_STATIC ai_intq_info_list conv2d_24_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #82 */
AI_STATIC ai_intq_info_list conv2d_25_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #83 */
AI_STATIC ai_intq_info_list conv2d_26_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.023528477177023888f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #84 */
AI_STATIC ai_intq_info_list conv2d_28_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.13508696854114532f), AI_PACK_UINTQ_ZP(95)));   /* Int quant #85 */
AI_STATIC ai_intq_info_list nl_30_fmt_output_intq = AI_INTQ_INFO_LIST_OBJ_INIT(
  AI_BUFFER_META_FLAG_SCALE_FLOAT|AI_BUFFER_META_FLAG_ZEROPOINT_U8, 1, AI_PACK_INTQ_INFO(
    AI_PACK_INTQ_SCALE(0.00390625f), AI_PACK_UINTQ_ZP(0)));   /* Int quant #86 */


/**  Tensor declarations section  *********************************************/
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &conv2d_28_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch1, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 5, 5), AI_STRIDE_INIT(4, 1, 1, 256, 1280),
  1, &conv2d_26_scratch1_array, &conv2d_26_scratch1_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1024, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1024, 1024),
  1, &conv2d_26_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 6913, 1, 1), AI_STRIDE_INIT(4, 1, 1, 6913, 6913),
  1, &conv2d_25_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_24_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_23_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_22_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_21_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_20_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_19_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_18_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_17_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_16_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_15_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 512, 1, 1), AI_STRIDE_INIT(4, 1, 1, 512, 512),
  1, &conv2d_14_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3457, 1, 1), AI_STRIDE_INIT(4, 1, 1, 3457, 3457),
  1, &conv2d_13_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &conv2d_12_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1729, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1729, 1729),
  1, &conv2d_11_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &conv2d_10_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1729, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1729, 1729),
  1, &conv2d_9_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_8_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 865, 1, 1), AI_STRIDE_INIT(4, 1, 1, 865, 865),
  1, &conv2d_7_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 1, 1, 128, 128),
  1, &conv2d_6_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 865, 1, 1), AI_STRIDE_INIT(4, 1, 1, 865, 865),
  1, &conv2d_5_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 1, 1, 64, 64),
  1, &conv2d_4_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 433, 1, 1), AI_STRIDE_INIT(4, 1, 1, 433, 433),
  1, &conv2d_3_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 1, 1, 32, 32),
  1, &conv2d_2_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 217, 1, 1), AI_STRIDE_INIT(4, 1, 1, 217, 217),
  1, &conv2d_1_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_scratch0, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 140, 1, 1), AI_STRIDE_INIT(4, 1, 1, 140, 140),
  1, &conv2d_0_scratch0_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1001, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4004, 4004),
  1, &conv2d_28_bias_array, &conv2d_28_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 256, 1, 1, 1001), AI_STRIDE_INIT(4, 1, 256, 256, 256),
  1, &conv2d_28_weights_array, &conv2d_28_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_26_bias_array, &conv2d_26_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 256, 1, 1, 256), AI_STRIDE_INIT(4, 1, 256, 256, 256),
  1, &conv2d_26_weights_array, &conv2d_26_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_25_bias_array, &conv2d_25_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 256, 3, 3, 1), AI_STRIDE_INIT(4, 1, 256, 768, 2304),
  1, &conv2d_25_weights_array, &conv2d_25_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 4, 4, 1024, 1024),
  1, &conv2d_24_bias_array, &conv2d_24_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 256), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_24_weights_array, &conv2d_24_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_23_bias_array, &conv2d_23_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_23_weights_array, &conv2d_23_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_22_bias_array, &conv2d_22_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_22_weights_array, &conv2d_22_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_21_bias_array, &conv2d_21_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_21_weights_array, &conv2d_21_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_20_bias_array, &conv2d_20_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_20_weights_array, &conv2d_20_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_19_bias_array, &conv2d_19_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_19_weights_array, &conv2d_19_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_18_bias_array, &conv2d_18_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_18_weights_array, &conv2d_18_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_17_bias_array, &conv2d_17_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_17_weights_array, &conv2d_17_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_16_bias_array, &conv2d_16_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_16_weights_array, &conv2d_16_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_15_bias_array, &conv2d_15_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_15_weights_array, &conv2d_15_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_14_bias_array, &conv2d_14_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 1, 1, 128), AI_STRIDE_INIT(4, 1, 128, 128, 128),
  1, &conv2d_14_weights_array, &conv2d_14_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_13_bias_array, &conv2d_13_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 128, 3, 3, 1), AI_STRIDE_INIT(4, 1, 128, 384, 1152),
  1, &conv2d_13_weights_array, &conv2d_13_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 1, 1), AI_STRIDE_INIT(4, 4, 4, 512, 512),
  1, &conv2d_12_bias_array, &conv2d_12_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 1, 1, 128), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_12_weights_array, &conv2d_12_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_11_bias_array, &conv2d_11_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 3, 3, 1), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_11_weights_array, &conv2d_11_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_10_bias_array, &conv2d_10_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 1, 1, 64), AI_STRIDE_INIT(4, 1, 64, 64, 64),
  1, &conv2d_10_weights_array, &conv2d_10_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_9_bias_array, &conv2d_9_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 64, 3, 3, 1), AI_STRIDE_INIT(4, 1, 64, 192, 576),
  1, &conv2d_9_weights_array, &conv2d_9_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 1, 1), AI_STRIDE_INIT(4, 4, 4, 256, 256),
  1, &conv2d_8_bias_array, &conv2d_8_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 1, 1, 64), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &conv2d_8_weights_array, &conv2d_8_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_7_bias_array, &conv2d_7_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 3, 3, 1), AI_STRIDE_INIT(4, 1, 32, 96, 288),
  1, &conv2d_7_weights_array, &conv2d_7_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_6_bias_array, &conv2d_6_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 1, 1, 32), AI_STRIDE_INIT(4, 1, 32, 32, 32),
  1, &conv2d_6_weights_array, &conv2d_6_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_5_bias_array, &conv2d_5_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 32, 3, 3, 1), AI_STRIDE_INIT(4, 1, 32, 96, 288),
  1, &conv2d_5_weights_array, &conv2d_5_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 1, 1), AI_STRIDE_INIT(4, 4, 4, 128, 128),
  1, &conv2d_4_bias_array, &conv2d_4_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 16, 1, 1, 32), AI_STRIDE_INIT(4, 1, 16, 16, 16),
  1, &conv2d_4_weights_array, &conv2d_4_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_3_bias_array, &conv2d_3_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 16, 3, 3, 1), AI_STRIDE_INIT(4, 1, 16, 48, 144),
  1, &conv2d_3_weights_array, &conv2d_3_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 1, 1), AI_STRIDE_INIT(4, 4, 4, 64, 64),
  1, &conv2d_2_bias_array, &conv2d_2_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 8, 1, 1, 16), AI_STRIDE_INIT(4, 1, 8, 8, 8),
  1, &conv2d_2_weights_array, &conv2d_2_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_1_bias_array, &conv2d_1_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 8, 3, 3, 1), AI_STRIDE_INIT(4, 1, 8, 24, 72),
  1, &conv2d_1_weights_array, &conv2d_1_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_bias, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 1, 1), AI_STRIDE_INIT(4, 4, 4, 32, 32),
  1, &conv2d_0_bias_array, &conv2d_0_bias_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_weights, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 3, 3, 3, 8), AI_STRIDE_INIT(4, 1, 3, 9, 27),
  1, &conv2d_0_weights_array, &conv2d_0_weights_intq)
AI_TENSOR_OBJ_DECLARE(
  input_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 3, 160, 160), AI_STRIDE_INIT(4, 1, 1, 3, 480),
  1, &input_0_output_array, &input_0_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_0_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_0_output_array, &conv2d_0_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_1_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 8, 80, 80), AI_STRIDE_INIT(4, 1, 1, 8, 640),
  1, &conv2d_1_output_array, &conv2d_1_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_2_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 80, 80), AI_STRIDE_INIT(4, 1, 1, 16, 1280),
  1, &conv2d_2_output_array, &conv2d_2_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_3_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 16, 40, 40), AI_STRIDE_INIT(4, 1, 1, 16, 640),
  1, &conv2d_3_output_array, &conv2d_3_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_4_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 1, 1, 32, 1280),
  1, &conv2d_4_output_array, &conv2d_4_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_5_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 1, 1, 32, 1280),
  1, &conv2d_5_output_array, &conv2d_5_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_6_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 40, 40), AI_STRIDE_INIT(4, 1, 1, 32, 1280),
  1, &conv2d_6_output_array, &conv2d_6_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_7_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 32, 20, 20), AI_STRIDE_INIT(4, 1, 1, 32, 640),
  1, &conv2d_7_output_array, &conv2d_7_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_8_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_8_output_array, &conv2d_8_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_9_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_9_output_array, &conv2d_9_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_10_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 20, 20), AI_STRIDE_INIT(4, 1, 1, 64, 1280),
  1, &conv2d_10_output_array, &conv2d_10_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_11_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 64, 10, 10), AI_STRIDE_INIT(4, 1, 1, 64, 640),
  1, &conv2d_11_output_array, &conv2d_11_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_12_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_12_output_array, &conv2d_12_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_13_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_13_output_array, &conv2d_13_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_14_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_14_output_array, &conv2d_14_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_15_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_15_output_array, &conv2d_15_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_16_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_16_output_array, &conv2d_16_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_17_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_17_output_array, &conv2d_17_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_18_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_18_output_array, &conv2d_18_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_19_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_19_output_array, &conv2d_19_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_20_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_20_output_array, &conv2d_20_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_21_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_21_output_array, &conv2d_21_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_22_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 10, 10), AI_STRIDE_INIT(4, 1, 1, 128, 1280),
  1, &conv2d_22_output_array, &conv2d_22_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_23_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 128, 5, 5), AI_STRIDE_INIT(4, 1, 1, 128, 640),
  1, &conv2d_23_output_array, &conv2d_23_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_24_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 5, 5), AI_STRIDE_INIT(4, 1, 1, 256, 1280),
  1, &conv2d_24_output_array, &conv2d_24_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_25_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 5, 5), AI_STRIDE_INIT(4, 1, 1, 256, 1280),
  1, &conv2d_25_output_array, &conv2d_25_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_26_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 256, 1, 1), AI_STRIDE_INIT(4, 1, 1, 256, 256),
  1, &conv2d_26_output_array, &conv2d_26_output_intq)
AI_TENSOR_OBJ_DECLARE(
  conv2d_28_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1001, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1001, 1001),
  1, &conv2d_28_output_array, &conv2d_28_output_intq)
AI_TENSOR_OBJ_DECLARE(
  reshape_29_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1001, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4004, 4004),
  1, &reshape_29_fmt_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  nl_30_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1001, 1, 1), AI_STRIDE_INIT(4, 4, 4, 4004, 4004),
  1, &nl_30_output_array, NULL)
AI_TENSOR_OBJ_DECLARE(
  nl_30_fmt_output, AI_STATIC,
  0x0, 0x0, AI_SHAPE_INIT(4, 1, 1001, 1, 1), AI_STRIDE_INIT(4, 1, 1, 1001, 1001),
  1, &nl_30_fmt_output_array, &nl_30_fmt_output_intq)


/**  Layer declarations section  **********************************************/



AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_0_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&input_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_weights, &conv2d_0_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_0_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_0_layer, 0,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_1_layer, AI_STATIC,
  .tensors = &conv2d_0_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_1_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_0_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_weights, &conv2d_1_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_1_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_1_layer, 1,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_2_layer, AI_STATIC,
  .tensors = &conv2d_1_chain, 
  .groups = 8, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_2_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_1_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_weights, &conv2d_2_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_2_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_2_layer, 2,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_3_layer, AI_STATIC,
  .tensors = &conv2d_2_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_3_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_2_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_3_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_3_weights, &conv2d_3_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_3_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_3_layer, 3,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_4_layer, AI_STATIC,
  .tensors = &conv2d_3_chain, 
  .groups = 16, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_4_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_3_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_weights, &conv2d_4_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_4_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_4_layer, 4,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_5_layer, AI_STATIC,
  .tensors = &conv2d_4_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_5_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_4_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_weights, &conv2d_5_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_5_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_5_layer, 5,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_6_layer, AI_STATIC,
  .tensors = &conv2d_5_chain, 
  .groups = 32, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_6_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_5_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_weights, &conv2d_6_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_6_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_6_layer, 6,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_7_layer, AI_STATIC,
  .tensors = &conv2d_6_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_7_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_6_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_7_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_7_weights, &conv2d_7_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_7_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_7_layer, 7,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_8_layer, AI_STATIC,
  .tensors = &conv2d_7_chain, 
  .groups = 32, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_8_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_7_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_weights, &conv2d_8_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_8_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_8_layer, 8,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_9_layer, AI_STATIC,
  .tensors = &conv2d_8_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_9_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_8_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_9_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_9_weights, &conv2d_9_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_9_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_9_layer, 9,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_10_layer, AI_STATIC,
  .tensors = &conv2d_9_chain, 
  .groups = 64, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_10_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_9_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_10_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_10_weights, &conv2d_10_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_10_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_10_layer, 10,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_11_layer, AI_STATIC,
  .tensors = &conv2d_10_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_11_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_10_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_11_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_11_weights, &conv2d_11_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_11_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_11_layer, 11,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_12_layer, AI_STATIC,
  .tensors = &conv2d_11_chain, 
  .groups = 64, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_12_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_11_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_12_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_12_weights, &conv2d_12_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_12_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_12_layer, 12,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_13_layer, AI_STATIC,
  .tensors = &conv2d_12_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_13_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_12_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_13_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_13_weights, &conv2d_13_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_13_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_13_layer, 13,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_14_layer, AI_STATIC,
  .tensors = &conv2d_13_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_14_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_13_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_14_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_14_weights, &conv2d_14_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_14_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_14_layer, 14,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_15_layer, AI_STATIC,
  .tensors = &conv2d_14_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_15_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_14_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_15_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_15_weights, &conv2d_15_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_15_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_15_layer, 15,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_16_layer, AI_STATIC,
  .tensors = &conv2d_15_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_16_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_15_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_16_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_16_weights, &conv2d_16_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_16_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_16_layer, 16,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_17_layer, AI_STATIC,
  .tensors = &conv2d_16_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_17_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_16_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_17_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_17_weights, &conv2d_17_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_17_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_17_layer, 17,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_18_layer, AI_STATIC,
  .tensors = &conv2d_17_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_18_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_17_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_18_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_18_weights, &conv2d_18_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_18_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_18_layer, 18,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_19_layer, AI_STATIC,
  .tensors = &conv2d_18_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_19_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_18_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_19_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_19_weights, &conv2d_19_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_19_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_19_layer, 19,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_20_layer, AI_STATIC,
  .tensors = &conv2d_19_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_20_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_19_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_20_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_20_weights, &conv2d_20_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_20_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_20_layer, 20,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_21_layer, AI_STATIC,
  .tensors = &conv2d_20_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_21_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_20_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_21_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_21_weights, &conv2d_21_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_21_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_21_layer, 21,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_22_layer, AI_STATIC,
  .tensors = &conv2d_21_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_22_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_21_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_22_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_22_weights, &conv2d_22_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_22_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_22_layer, 22,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_23_layer, AI_STATIC,
  .tensors = &conv2d_22_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_23_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_22_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_23_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_23_weights, &conv2d_23_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_23_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_23_layer, 23,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_24_layer, AI_STATIC,
  .tensors = &conv2d_23_chain, 
  .groups = 128, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(2, 2), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 2, 2), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_24_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_23_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_24_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_24_weights, &conv2d_24_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_24_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_24_layer, 24,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_25_layer, AI_STATIC,
  .tensors = &conv2d_24_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_25_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_24_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_25_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_25_weights, &conv2d_25_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_25_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_25_layer, 25,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_26_layer, AI_STATIC,
  .tensors = &conv2d_25_chain, 
  .groups = 256, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 1, 1, 1, 1), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_26_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_25_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_26_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_26_weights, &conv2d_26_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_26_scratch0, &conv2d_26_scratch1)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_26_layer, 26,
  OPTIMIZED_CONV2D_TYPE,
  conv2d_nl_pool, forward_conv2d_nl_pool_integer,
  &AI_NET_OBJ_INSTANCE, &conv2d_28_layer, AI_STATIC,
  .tensors = &conv2d_26_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_size = AI_SHAPE_2D_INIT(5, 5), 
  .pool_stride = AI_SHAPE_2D_INIT(2, 2), 
  .pool_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
  .pool_func = pool_func_ap_array_integer, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  conv2d_28_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_26_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_28_output),
  AI_TENSOR_LIST_ENTRY(&conv2d_28_weights, &conv2d_28_bias, NULL),
  AI_TENSOR_LIST_ENTRY(&conv2d_28_scratch0)
)

AI_LAYER_OBJ_DECLARE(
  conv2d_28_layer, 28,
  CONV2D_TYPE,
  conv2d, forward_conv2d_integer,
  &AI_NET_OBJ_INSTANCE, &reshape_29_fmt_layer, AI_STATIC,
  .tensors = &conv2d_28_chain, 
  .groups = 1, 
  .nl_func = NULL, 
  .filter_stride = AI_SHAPE_2D_INIT(1, 1), 
  .dilation = AI_SHAPE_2D_INIT(1, 1), 
  .filter_pad = AI_SHAPE_INIT(4, 0, 0, 0, 0), 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  reshape_29_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&conv2d_28_output),
  AI_TENSOR_LIST_ENTRY(&reshape_29_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  reshape_29_fmt_layer, 29,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &nl_30_layer, AI_STATIC,
  .tensors = &reshape_29_fmt_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_30_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&reshape_29_fmt_output),
  AI_TENSOR_LIST_ENTRY(&nl_30_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_30_layer, 30,
  NL_TYPE,
  nl, forward_sm,
  &AI_NET_OBJ_INSTANCE, &nl_30_fmt_layer, AI_STATIC,
  .tensors = &nl_30_chain, 
)

AI_TENSOR_CHAIN_OBJ_DECLARE(
  nl_30_fmt_chain, AI_STATIC_CONST, 4,
  AI_TENSOR_LIST_ENTRY(&nl_30_output),
  AI_TENSOR_LIST_ENTRY(&nl_30_fmt_output),
  AI_TENSOR_LIST_EMPTY,
  AI_TENSOR_LIST_EMPTY
)

AI_LAYER_OBJ_DECLARE(
  nl_30_fmt_layer, 30,
  NL_TYPE,
  nl, node_convert,
  &AI_NET_OBJ_INSTANCE, &nl_30_fmt_layer, AI_STATIC,
  .tensors = &nl_30_fmt_chain, 
)


AI_NETWORK_OBJ_DECLARE(
  AI_NET_OBJ_INSTANCE, AI_STATIC,
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 478804, 1,
                     NULL),
  AI_BUFFER_OBJ_INIT(AI_BUFFER_FORMAT_U8,
                     1, 1, 205160, 1,
                     NULL),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_CNN_IN_NUM, &input_0_output),
  AI_TENSOR_LIST_IO_ENTRY(AI_FLAG_NONE, AI_CNN_OUT_NUM, &nl_30_fmt_output),
  &conv2d_0_layer, 0, NULL)



AI_DECLARE_STATIC
ai_bool cnn_configure_activations(
  ai_network* net_ctx, const ai_buffer* activation_buffer)
{
  AI_ASSERT(net_ctx &&  activation_buffer && activation_buffer->data)

  ai_ptr activations = AI_PTR(AI_PTR_ALIGN(activation_buffer->data, 4));
  AI_ASSERT(activations)
  AI_UNUSED(net_ctx)

  {
    /* Updating activations (byte) offsets */
    conv2d_28_scratch0_array.data = AI_PTR(activations + 27332);
    conv2d_28_scratch0_array.data_start = AI_PTR(activations + 27332);
    conv2d_26_scratch1_array.data = AI_PTR(activations + 9860);
    conv2d_26_scratch1_array.data_start = AI_PTR(activations + 9860);
    conv2d_26_scratch0_array.data = AI_PTR(activations + 27332);
    conv2d_26_scratch0_array.data_start = AI_PTR(activations + 27332);
    conv2d_25_scratch0_array.data = AI_PTR(activations + 42384);
    conv2d_25_scratch0_array.data_start = AI_PTR(activations + 42384);
    conv2d_24_scratch0_array.data = AI_PTR(activations + 26640);
    conv2d_24_scratch0_array.data_start = AI_PTR(activations + 26640);
    conv2d_23_scratch0_array.data = AI_PTR(activations + 32524);
    conv2d_23_scratch0_array.data_start = AI_PTR(activations + 32524);
    conv2d_22_scratch0_array.data = AI_PTR(activations + 26640);
    conv2d_22_scratch0_array.data_start = AI_PTR(activations + 26640);
    conv2d_21_scratch0_array.data = AI_PTR(activations + 23180);
    conv2d_21_scratch0_array.data_start = AI_PTR(activations + 23180);
    conv2d_20_scratch0_array.data = AI_PTR(activations + 51996);
    conv2d_20_scratch0_array.data_start = AI_PTR(activations + 51996);
    conv2d_19_scratch0_array.data = AI_PTR(activations + 19720);
    conv2d_19_scratch0_array.data_start = AI_PTR(activations + 19720);
    conv2d_18_scratch0_array.data = AI_PTR(activations + 51996);
    conv2d_18_scratch0_array.data_start = AI_PTR(activations + 51996);
    conv2d_17_scratch0_array.data = AI_PTR(activations + 16260);
    conv2d_17_scratch0_array.data_start = AI_PTR(activations + 16260);
    conv2d_16_scratch0_array.data = AI_PTR(activations + 51996);
    conv2d_16_scratch0_array.data_start = AI_PTR(activations + 51996);
    conv2d_15_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_15_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_14_scratch0_array.data = AI_PTR(activations + 51996);
    conv2d_14_scratch0_array.data_start = AI_PTR(activations + 51996);
    conv2d_13_scratch0_array.data = AI_PTR(activations + 29064);
    conv2d_13_scratch0_array.data_start = AI_PTR(activations + 29064);
    conv2d_12_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_12_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_11_scratch0_array.data = AI_PTR(activations + 27332);
    conv2d_11_scratch0_array.data_start = AI_PTR(activations + 27332);
    conv2d_10_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_10_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_9_scratch0_array.data = AI_PTR(activations + 25600);
    conv2d_9_scratch0_array.data_start = AI_PTR(activations + 25600);
    conv2d_8_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_8_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_7_scratch0_array.data = AI_PTR(activations + 52864);
    conv2d_7_scratch0_array.data_start = AI_PTR(activations + 52864);
    conv2d_6_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_6_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_5_scratch0_array.data = AI_PTR(activations + 51996);
    conv2d_5_scratch0_array.data_start = AI_PTR(activations + 51996);
    conv2d_4_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_4_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_3_scratch0_array.data = AI_PTR(activations + 51560);
    conv2d_3_scratch0_array.data_start = AI_PTR(activations + 51560);
    conv2d_2_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_2_scratch0_array.data_start = AI_PTR(activations + 0);
    conv2d_1_scratch0_array.data = AI_PTR(activations + 51340);
    conv2d_1_scratch0_array.data_start = AI_PTR(activations + 51340);
    conv2d_0_scratch0_array.data = AI_PTR(activations + 0);
    conv2d_0_scratch0_array.data_start = AI_PTR(activations + 0);
    input_0_output_array.data = AI_PTR(NULL);
    input_0_output_array.data_start = AI_PTR(NULL);
    conv2d_0_output_array.data = AI_PTR(activations + 140);
    conv2d_0_output_array.data_start = AI_PTR(activations + 140);
    conv2d_1_output_array.data = AI_PTR(activations + 51560);
    conv2d_1_output_array.data_start = AI_PTR(activations + 51560);
    conv2d_2_output_array.data = AI_PTR(activations + 102760);
    conv2d_2_output_array.data_start = AI_PTR(activations + 102760);
    conv2d_3_output_array.data = AI_PTR(activations + 51996);
    conv2d_3_output_array.data_start = AI_PTR(activations + 51996);
    conv2d_4_output_array.data = AI_PTR(activations + 0);
    conv2d_4_output_array.data_start = AI_PTR(activations + 0);
    conv2d_5_output_array.data = AI_PTR(activations + 52864);
    conv2d_5_output_array.data_start = AI_PTR(activations + 52864);
    conv2d_6_output_array.data = AI_PTR(activations + 0);
    conv2d_6_output_array.data_start = AI_PTR(activations + 0);
    conv2d_7_output_array.data = AI_PTR(activations + 53732);
    conv2d_7_output_array.data_start = AI_PTR(activations + 53732);
    conv2d_8_output_array.data = AI_PTR(activations + 0);
    conv2d_8_output_array.data_start = AI_PTR(activations + 0);
    conv2d_9_output_array.data = AI_PTR(activations + 53732);
    conv2d_9_output_array.data_start = AI_PTR(activations + 53732);
    conv2d_10_output_array.data = AI_PTR(activations + 0);
    conv2d_10_output_array.data_start = AI_PTR(activations + 0);
    conv2d_11_output_array.data = AI_PTR(activations + 29064);
    conv2d_11_output_array.data_start = AI_PTR(activations + 29064);
    conv2d_12_output_array.data = AI_PTR(activations + 35464);
    conv2d_12_output_array.data_start = AI_PTR(activations + 35464);
    conv2d_13_output_array.data = AI_PTR(activations + 0);
    conv2d_13_output_array.data_start = AI_PTR(activations + 0);
    conv2d_14_output_array.data = AI_PTR(activations + 12800);
    conv2d_14_output_array.data_start = AI_PTR(activations + 12800);
    conv2d_15_output_array.data = AI_PTR(activations + 32524);
    conv2d_15_output_array.data_start = AI_PTR(activations + 32524);
    conv2d_16_output_array.data = AI_PTR(activations + 3460);
    conv2d_16_output_array.data_start = AI_PTR(activations + 3460);
    conv2d_17_output_array.data = AI_PTR(activations + 32524);
    conv2d_17_output_array.data_start = AI_PTR(activations + 32524);
    conv2d_18_output_array.data = AI_PTR(activations + 3460);
    conv2d_18_output_array.data_start = AI_PTR(activations + 3460);
    conv2d_19_output_array.data = AI_PTR(activations + 32524);
    conv2d_19_output_array.data_start = AI_PTR(activations + 32524);
    conv2d_20_output_array.data = AI_PTR(activations + 3460);
    conv2d_20_output_array.data_start = AI_PTR(activations + 3460);
    conv2d_21_output_array.data = AI_PTR(activations + 32524);
    conv2d_21_output_array.data_start = AI_PTR(activations + 32524);
    conv2d_22_output_array.data = AI_PTR(activations + 3460);
    conv2d_22_output_array.data_start = AI_PTR(activations + 3460);
    conv2d_23_output_array.data = AI_PTR(activations + 0);
    conv2d_23_output_array.data_start = AI_PTR(activations + 0);
    conv2d_24_output_array.data = AI_PTR(activations + 35984);
    conv2d_24_output_array.data_start = AI_PTR(activations + 35984);
    conv2d_25_output_array.data = AI_PTR(activations + 3460);
    conv2d_25_output_array.data_start = AI_PTR(activations + 3460);
    conv2d_26_output_array.data = AI_PTR(activations + 51560);
    conv2d_26_output_array.data_start = AI_PTR(activations + 51560);
    conv2d_28_output_array.data = AI_PTR(activations + 49300);
    conv2d_28_output_array.data_start = AI_PTR(activations + 49300);
    reshape_29_fmt_output_array.data = AI_PTR(activations + 35984);
    reshape_29_fmt_output_array.data_start = AI_PTR(activations + 35984);
    nl_30_output_array.data = AI_PTR(activations + 35984);
    nl_30_output_array.data_start = AI_PTR(activations + 35984);
    nl_30_fmt_output_array.data = AI_PTR(NULL);
    nl_30_fmt_output_array.data_start = AI_PTR(NULL);
    
  }
  return true;
}



AI_DECLARE_STATIC
ai_bool cnn_configure_weights(
  ai_network* net_ctx, const ai_buffer* weights_buffer)
{
  AI_ASSERT(net_ctx &&  weights_buffer && weights_buffer->data)

  ai_ptr weights = AI_PTR(weights_buffer->data);
  AI_ASSERT(weights)
  AI_UNUSED(net_ctx)

  {
    /* Updating weights (byte) offsets */
    
    conv2d_28_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_bias_array.data = AI_PTR(weights + 474800);
    conv2d_28_bias_array.data_start = AI_PTR(weights + 474800);
    conv2d_28_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_28_weights_array.data = AI_PTR(weights + 218544);
    conv2d_28_weights_array.data_start = AI_PTR(weights + 218544);
    conv2d_26_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_bias_array.data = AI_PTR(weights + 217520);
    conv2d_26_bias_array.data_start = AI_PTR(weights + 217520);
    conv2d_26_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_26_weights_array.data = AI_PTR(weights + 151984);
    conv2d_26_weights_array.data_start = AI_PTR(weights + 151984);
    conv2d_25_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_25_bias_array.data = AI_PTR(weights + 150960);
    conv2d_25_bias_array.data_start = AI_PTR(weights + 150960);
    conv2d_25_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_25_weights_array.data = AI_PTR(weights + 148656);
    conv2d_25_weights_array.data_start = AI_PTR(weights + 148656);
    conv2d_24_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_bias_array.data = AI_PTR(weights + 147632);
    conv2d_24_bias_array.data_start = AI_PTR(weights + 147632);
    conv2d_24_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_24_weights_array.data = AI_PTR(weights + 114864);
    conv2d_24_weights_array.data_start = AI_PTR(weights + 114864);
    conv2d_23_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_23_bias_array.data = AI_PTR(weights + 114352);
    conv2d_23_bias_array.data_start = AI_PTR(weights + 114352);
    conv2d_23_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_23_weights_array.data = AI_PTR(weights + 113200);
    conv2d_23_weights_array.data_start = AI_PTR(weights + 113200);
    conv2d_22_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_bias_array.data = AI_PTR(weights + 112688);
    conv2d_22_bias_array.data_start = AI_PTR(weights + 112688);
    conv2d_22_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_22_weights_array.data = AI_PTR(weights + 96304);
    conv2d_22_weights_array.data_start = AI_PTR(weights + 96304);
    conv2d_21_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_bias_array.data = AI_PTR(weights + 95792);
    conv2d_21_bias_array.data_start = AI_PTR(weights + 95792);
    conv2d_21_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_21_weights_array.data = AI_PTR(weights + 94640);
    conv2d_21_weights_array.data_start = AI_PTR(weights + 94640);
    conv2d_20_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_bias_array.data = AI_PTR(weights + 94128);
    conv2d_20_bias_array.data_start = AI_PTR(weights + 94128);
    conv2d_20_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_20_weights_array.data = AI_PTR(weights + 77744);
    conv2d_20_weights_array.data_start = AI_PTR(weights + 77744);
    conv2d_19_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_bias_array.data = AI_PTR(weights + 77232);
    conv2d_19_bias_array.data_start = AI_PTR(weights + 77232);
    conv2d_19_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_19_weights_array.data = AI_PTR(weights + 76080);
    conv2d_19_weights_array.data_start = AI_PTR(weights + 76080);
    conv2d_18_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_18_bias_array.data = AI_PTR(weights + 75568);
    conv2d_18_bias_array.data_start = AI_PTR(weights + 75568);
    conv2d_18_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_18_weights_array.data = AI_PTR(weights + 59184);
    conv2d_18_weights_array.data_start = AI_PTR(weights + 59184);
    conv2d_17_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_bias_array.data = AI_PTR(weights + 58672);
    conv2d_17_bias_array.data_start = AI_PTR(weights + 58672);
    conv2d_17_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_17_weights_array.data = AI_PTR(weights + 57520);
    conv2d_17_weights_array.data_start = AI_PTR(weights + 57520);
    conv2d_16_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_bias_array.data = AI_PTR(weights + 57008);
    conv2d_16_bias_array.data_start = AI_PTR(weights + 57008);
    conv2d_16_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_16_weights_array.data = AI_PTR(weights + 40624);
    conv2d_16_weights_array.data_start = AI_PTR(weights + 40624);
    conv2d_15_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_bias_array.data = AI_PTR(weights + 40112);
    conv2d_15_bias_array.data_start = AI_PTR(weights + 40112);
    conv2d_15_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_15_weights_array.data = AI_PTR(weights + 38960);
    conv2d_15_weights_array.data_start = AI_PTR(weights + 38960);
    conv2d_14_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_bias_array.data = AI_PTR(weights + 38448);
    conv2d_14_bias_array.data_start = AI_PTR(weights + 38448);
    conv2d_14_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_14_weights_array.data = AI_PTR(weights + 22064);
    conv2d_14_weights_array.data_start = AI_PTR(weights + 22064);
    conv2d_13_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_bias_array.data = AI_PTR(weights + 21552);
    conv2d_13_bias_array.data_start = AI_PTR(weights + 21552);
    conv2d_13_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_13_weights_array.data = AI_PTR(weights + 20400);
    conv2d_13_weights_array.data_start = AI_PTR(weights + 20400);
    conv2d_12_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_bias_array.data = AI_PTR(weights + 19888);
    conv2d_12_bias_array.data_start = AI_PTR(weights + 19888);
    conv2d_12_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_12_weights_array.data = AI_PTR(weights + 11696);
    conv2d_12_weights_array.data_start = AI_PTR(weights + 11696);
    conv2d_11_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_bias_array.data = AI_PTR(weights + 11440);
    conv2d_11_bias_array.data_start = AI_PTR(weights + 11440);
    conv2d_11_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_11_weights_array.data = AI_PTR(weights + 10864);
    conv2d_11_weights_array.data_start = AI_PTR(weights + 10864);
    conv2d_10_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_bias_array.data = AI_PTR(weights + 10608);
    conv2d_10_bias_array.data_start = AI_PTR(weights + 10608);
    conv2d_10_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_10_weights_array.data = AI_PTR(weights + 6512);
    conv2d_10_weights_array.data_start = AI_PTR(weights + 6512);
    conv2d_9_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_bias_array.data = AI_PTR(weights + 6256);
    conv2d_9_bias_array.data_start = AI_PTR(weights + 6256);
    conv2d_9_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_9_weights_array.data = AI_PTR(weights + 5680);
    conv2d_9_weights_array.data_start = AI_PTR(weights + 5680);
    conv2d_8_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_bias_array.data = AI_PTR(weights + 5424);
    conv2d_8_bias_array.data_start = AI_PTR(weights + 5424);
    conv2d_8_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_8_weights_array.data = AI_PTR(weights + 3376);
    conv2d_8_weights_array.data_start = AI_PTR(weights + 3376);
    conv2d_7_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_bias_array.data = AI_PTR(weights + 3248);
    conv2d_7_bias_array.data_start = AI_PTR(weights + 3248);
    conv2d_7_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_7_weights_array.data = AI_PTR(weights + 2960);
    conv2d_7_weights_array.data_start = AI_PTR(weights + 2960);
    conv2d_6_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_bias_array.data = AI_PTR(weights + 2832);
    conv2d_6_bias_array.data_start = AI_PTR(weights + 2832);
    conv2d_6_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_6_weights_array.data = AI_PTR(weights + 1808);
    conv2d_6_weights_array.data_start = AI_PTR(weights + 1808);
    conv2d_5_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_bias_array.data = AI_PTR(weights + 1680);
    conv2d_5_bias_array.data_start = AI_PTR(weights + 1680);
    conv2d_5_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_5_weights_array.data = AI_PTR(weights + 1392);
    conv2d_5_weights_array.data_start = AI_PTR(weights + 1392);
    conv2d_4_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_bias_array.data = AI_PTR(weights + 1264);
    conv2d_4_bias_array.data_start = AI_PTR(weights + 1264);
    conv2d_4_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_4_weights_array.data = AI_PTR(weights + 752);
    conv2d_4_weights_array.data_start = AI_PTR(weights + 752);
    conv2d_3_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_bias_array.data = AI_PTR(weights + 688);
    conv2d_3_bias_array.data_start = AI_PTR(weights + 688);
    conv2d_3_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_3_weights_array.data = AI_PTR(weights + 544);
    conv2d_3_weights_array.data_start = AI_PTR(weights + 544);
    conv2d_2_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_bias_array.data = AI_PTR(weights + 480);
    conv2d_2_bias_array.data_start = AI_PTR(weights + 480);
    conv2d_2_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_2_weights_array.data = AI_PTR(weights + 352);
    conv2d_2_weights_array.data_start = AI_PTR(weights + 352);
    conv2d_1_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_bias_array.data = AI_PTR(weights + 320);
    conv2d_1_bias_array.data_start = AI_PTR(weights + 320);
    conv2d_1_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_1_weights_array.data = AI_PTR(weights + 248);
    conv2d_1_weights_array.data_start = AI_PTR(weights + 248);
    conv2d_0_bias_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_bias_array.data = AI_PTR(weights + 216);
    conv2d_0_bias_array.data_start = AI_PTR(weights + 216);
    conv2d_0_weights_array.format |= AI_FMT_FLAG_CONST;
    conv2d_0_weights_array.data = AI_PTR(weights + 0);
    conv2d_0_weights_array.data_start = AI_PTR(weights + 0);
  }

  return true;
}


/**  PUBLIC APIs SECTION  *****************************************************/

AI_API_ENTRY
ai_bool ai_cnn_get_info(
  ai_handle network, ai_network_report* report)
{
  ai_network* net_ctx = AI_NETWORK_ACQUIRE_CTX(network);

  if ( report && net_ctx )
  {
    ai_network_report r = {
      .model_name        = AI_CNN_MODEL_NAME,
      .model_signature   = AI_CNN_MODEL_SIGNATURE,
      .model_datetime    = AI_TOOLS_DATE_TIME,
      
      .compile_datetime  = AI_TOOLS_COMPILE_TIME,
      
      .runtime_revision  = ai_platform_runtime_get_revision(),
      .runtime_version   = ai_platform_runtime_get_version(),

      .tool_revision     = AI_TOOLS_REVISION_ID,
      .tool_version      = {AI_TOOLS_VERSION_MAJOR, AI_TOOLS_VERSION_MINOR,
                            AI_TOOLS_VERSION_MICRO, 0x0},
      .tool_api_version  = {AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR,
                            AI_TOOLS_API_VERSION_MICRO, 0x0},

      .api_version            = ai_platform_api_get_version(),
      .interface_api_version  = ai_platform_interface_api_get_version(),
      
      .n_macc            = 21088612,
      .n_inputs          = 0,
      .inputs            = NULL,
      .n_outputs         = 0,
      .outputs           = NULL,
      .activations       = AI_STRUCT_INIT,
      .params            = AI_STRUCT_INIT,
      .n_nodes           = 0,
      .signature         = 0x0,
    };

    if ( !ai_platform_api_get_network_report(network, &r) ) return false;

    *report = r;
    return true;
  }

  return false;
}

AI_API_ENTRY
ai_error ai_cnn_get_error(ai_handle network)
{
  return ai_platform_network_get_error(network);
}

AI_API_ENTRY
ai_error ai_cnn_create(
  ai_handle* network, const ai_buffer* network_config)
{
  return ai_platform_network_create(
    network, network_config, 
    &AI_NET_OBJ_INSTANCE,
    AI_TOOLS_API_VERSION_MAJOR, AI_TOOLS_API_VERSION_MINOR, AI_TOOLS_API_VERSION_MICRO);
}

AI_API_ENTRY
ai_handle ai_cnn_destroy(ai_handle network)
{
  return ai_platform_network_destroy(network);
}

AI_API_ENTRY
ai_bool ai_cnn_init(
  ai_handle network, const ai_network_params* params)
{
  ai_network* net_ctx = ai_platform_network_init(network, params);
  if ( !net_ctx ) return false;

  ai_bool ok = true;
  ok &= cnn_configure_weights(net_ctx, &params->params);
  ok &= cnn_configure_activations(net_ctx, &params->activations);

  return ok;
}


AI_API_ENTRY
ai_i32 ai_cnn_run(
  ai_handle network, const ai_buffer* input, ai_buffer* output)
{
  return ai_platform_network_process(network, input, output);
}

AI_API_ENTRY
ai_i32 ai_cnn_forward(ai_handle network, const ai_buffer* input)
{
  return ai_platform_network_process(network, input, NULL);
}

#undef AI_CNN_MODEL_SIGNATURE
#undef AI_NET_OBJ_INSTANCE
#undef AI_TOOLS_VERSION_MAJOR
#undef AI_TOOLS_VERSION_MINOR
#undef AI_TOOLS_VERSION_MICRO
#undef AI_TOOLS_API_VERSION_MAJOR
#undef AI_TOOLS_API_VERSION_MINOR
#undef AI_TOOLS_API_VERSION_MICRO
#undef AI_TOOLS_DATE_TIME
#undef AI_TOOLS_COMPILE_TIME

