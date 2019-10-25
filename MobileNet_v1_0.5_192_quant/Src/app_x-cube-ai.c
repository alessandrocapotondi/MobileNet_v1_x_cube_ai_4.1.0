#ifdef __cplusplus
 extern "C" {
#endif
/**
  ******************************************************************************
  * @file           : app_x-cube-ai.c
  * @brief          : AI program body
  ******************************************************************************
  * This notice applies to any and all portions of this file
  * that are not between comment pairs USER CODE BEGIN and
  * USER CODE END. Other portions of this file, whether
  * inserted by the user or by software development tools
  * are owned by their respective copyright owners.
  *
  * Copyright (c) 2018 STMicroelectronics International N.V.
  * All rights reserved.
  *
  * Redistribution and use in source and binary forms, with or without
  * modification, are permitted, provided that the following conditions are met:
  *
  * 1. Redistribution of source code must retain the above copyright notice,
  *    this list of conditions and the following disclaimer.
  * 2. Redistributions in binary form must reproduce the above copyright notice,
  *    this list of conditions and the following disclaimer in the documentation
  *    and/or other materials provided with the distribution.
  * 3. Neither the name of STMicroelectronics nor the names of other
  *    contributors to this software may be used to endorse or promote products
  *    derived from this software without specific written permission.
  * 4. This software, including modifications and/or derivative works of this
  *    software, must execute solely and exclusively on microcontroller or
  *    microprocessor devices manufactured by or for STMicroelectronics.
  * 5. Redistribution and use of this software other than as permitted under
  *    this license is void and will automatically terminate your rights under
  *    this license.
  *
  * THIS SOFTWARE IS PROVIDED BY STMICROELECTRONICS AND CONTRIBUTORS "AS IS"
  * AND ANY EXPRESS, IMPLIED OR STATUTORY WARRANTIES, INCLUDING, BUT NOT
  * LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY, FITNESS FOR A
  * PARTICULAR PURPOSE AND NON-INFRINGEMENT OF THIRD PARTY INTELLECTUAL PROPERTY
  * RIGHTS ARE DISCLAIMED TO THE FULLEST EXTENT PERMITTED BY LAW. IN NO EVENT
  * SHALL STMICROELECTRONICS OR CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT,
  * INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT
  * LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA,
  * OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
  * LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
  * NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS SOFTWARE,
  * EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
  *
  ******************************************************************************
  */
/* Includes ------------------------------------------------------------------*/
#include <string.h>
#include "app_x-cube-ai.h"
#include "main.h"
#include "ai_datatypes_defines.h"

/* USER CODE BEGIN includes */
#if __APP_X_PROFILE__
volatile unsigned int *DWT_CYCCNT = (volatile unsigned int *)0xE0001004; // Cycle Counter Register
volatile unsigned int *DWT_CONTROL= (volatile unsigned int *)0xE0001000; // Counter Control Register
volatile unsigned int *SCB_DEMCR  = (volatile unsigned int *)0xE000EDFC;

long long usr_nr_nb_run = 0LL;
long long usr_cycle_counter = 0LL;
long long usr_cycle_counter_acc = 0LL;
long long usr_cycle_counter_avg = 0LL;
#endif

/* Network Handle */
ai_handle ai_cnn_network = AI_HANDLE_NULL;

/* Global buffer to handle the activations data buffer - R/W data */
AI_ALIGNED(4)
ai_u8 __attribute__((section (".Activations"))) ai_cnn_activations[AI_CNN_DATA_ACTIVATIONS_SIZE];

/* Buffers to store the tensor input/output of the network */
ai_u8 __attribute__((section (".IOBuffers"))) in_data[AI_CNN_IN_1_SIZE];
ai_u8 __attribute__((section (".IOBuffers"))) out_data[AI_CNN_OUT_1_SIZE];
/* USER CODE END includes */

/*************************************************************************
  *
  */
void MX_X_CUBE_AI_Init(void)
{
    /* USER CODE BEGIN 0 */
	const ai_network_params params = {
			AI_CNN_DATA_WEIGHTS(ai_cnn_data_weights_get()),
			AI_CNN_DATA_ACTIVATIONS(ai_cnn_activations) };

	ai_error err = ai_cnn_create(&ai_cnn_network, AI_CNN_DATA_CONFIG);
	assert(err.type==AI_ERROR_NONE);

	ai_bool ret = ai_cnn_init(ai_cnn_network, &params);
	assert(ret==1);
    /* USER CODE END 0 */
}

void MX_X_CUBE_AI_Process(void)
{
    /* USER CODE BEGIN 1 */
	ai_i32 nbatch;

	/* Parameters checking */
	assert (ai_cnn_network);

	/* AI buffer handlers */
	ai_buffer ai_input[AI_CNN_IN_NUM] = AI_CNN_IN;
	ai_buffer ai_output[AI_CNN_OUT_NUM] = AI_CNN_OUT;

	/* Initialize input/output buffer handlers */
	ai_input[0].n_batches = 1;
	ai_input[0].data = AI_HANDLE_PTR(in_data);
	ai_output[0].n_batches = 1;
	ai_output[0].data = AI_HANDLE_PTR(out_data);

	/* Perform the inference */
	nbatch = ai_cnn_run(ai_cnn_network, &ai_input[0], &ai_output[0]);

	assert(nbatch==1);
    /* USER CODE END 1 */
}
#ifdef __cplusplus
}
#endif
