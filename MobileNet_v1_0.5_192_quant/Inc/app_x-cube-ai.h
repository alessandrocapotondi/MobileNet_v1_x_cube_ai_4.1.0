
/* Define to prevent recursive inclusion -------------------------------------*/
#ifndef __APP_AI_H
#define __APP_AI_H
#ifdef __cplusplus
 extern "C" {
#endif
/**
  ******************************************************************************
  * @file           : app_x-cube-ai.h
  * @brief          : AI entry function definitions
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
#include "ai_platform.h"
#include "cnn.h"
#include "cnn_data.h"

#define MIN_HEAP_SIZE 0x0
#define MIN_STACK_SIZE 0x400

#define AI_CNN_DATA_ACTIVATIONS_START_ADDR 0xFFFFFFFF

#define AI_MNETWORK_DATA_ACTIVATIONS_INT_SIZE AI_CNN_DATA_ACTIVATIONS_SIZE

void MX_X_CUBE_AI_Init(void);
void MX_X_CUBE_AI_Process(void);
/* USER CODE BEGIN includes */

#include <assert.h>
/******** Cycle counter defines  **********/
#define __APP_X_PROFILE__ 1
#if __APP_X_PROFILE__
extern volatile unsigned int *DWT_CYCCNT;
extern volatile unsigned int *DWT_CONTROL;
extern volatile unsigned int *SCB_DEMCR;
extern long long usr_nr_nb_run;
extern long long usr_cycle_counter;
extern long long usr_cycle_counter_acc;
extern long long usr_cycle_counter_avg;

#define USR_MEM_BARRIER() \
	do { \
		asm volatile("" ::: "memory");\
	} while (0)

#define USR_CC_ENABLE() \
  do { \
	asm volatile("" ::: "memory"); \
	*SCB_DEMCR = *SCB_DEMCR | 0x01000000;\
	*DWT_CYCCNT = 0; \
	*DWT_CONTROL = *DWT_CONTROL | 1; \
	asm volatile("" ::: "memory"); \
  } while (0)

#define USR_CC_RESET() \
  do { \
	asm volatile("" ::: "memory"); \
	*DWT_CYCCNT = 0; \
	asm volatile("" ::: "memory"); \
  } while (0)

#define USR_GET_CC_TIMESTAMP(x) \
  do { \
	asm volatile("" ::: "memory"); \
	x = (*(volatile unsigned int *) DWT_CYCCNT); \
	asm volatile("" ::: "memory"); \
  } while (0)

#define USR_GET_CC_ACC_TIMESTAMP(acc) \
  do { \
	asm volatile("" ::: "memory"); \
	acc += (*(volatile unsigned int *) DWT_CYCCNT); \
	asm volatile("" ::: "memory"); \
  } while (0)
#else
#define USR_MEM_BARRIER()
#define USR_CC_ENABLE()
#define USR_CC_RESET()
#define USR_GET_CC_TIMESTAMP(x)
#endif

/* Network Handle */
extern ai_handle ai_cnn_network;

/* Global buffer to handle the activations data buffer - R/W data */
extern ai_u8 ai_cnn_activations[];

/* Buffers to store the tensor input/output of the network */
extern ai_u8 in_data[];
extern ai_u8 out_data[];

/* USER CODE END includes */

#define AI_MNETWORK_NUMBER  (1)
//#define AI_MNETWORK_DATA_ACTIVATIONS_SIZE AI_CNN_DATA_ACTIVATIONS_SIZE
#define AI_MNETWORK_IN_1_SIZE AI_CNN_IN_1_SIZE
#define AI_MNETWORK_IN_1_SIZE_BYTES AI_CNN_IN_1_SIZE_BYTES
#define AI_MNETWORK_OUT_1_SIZE AI_CNN_OUT_1_SIZE
#define AI_MNETWORK_OUT_1_SIZE_BYTES AI_CNN_OUT_1_SIZE_BYTES
#ifdef __cplusplus
}
#endif

#endif /*__STMicroelectronics_X-CUBE-AI_4_1_0_H */
