/* Read-only model storage for the STM32F746G-DISCO NCNN smoke test. */
#include <stdio.h>

#include "stm32746g_discovery_qspi.h"

int ncnn_stm32f746_qspi_map(void) {
	QSPI_HandleTypeDef handle = {0};
	QSPI_CommandTypeDef command = {0};
	QSPI_MemoryMappedTypeDef mapping = {0};
	uint8_t id[3];

	/* Reuse the board's pin/clock setup, but not its Micron-only flash
	 * initialization: the tested board carries a Winbond W25Q128FV/JV. */
	handle.Instance = QUADSPI;
	if (HAL_QSPI_DeInit(&handle) != HAL_OK) {
		return -1;
	}
	BSP_QSPI_MspInit(&handle, NULL);
	handle.Init.ClockPrescaler = 7; /* 216 / 8 = 27 MHz, read 03h <= 50 MHz. */
	handle.Init.FifoThreshold = 4;
	handle.Init.SampleShifting = QSPI_SAMPLE_SHIFTING_HALFCYCLE;
	handle.Init.FlashSize = 23; /* 24 address bits, 16 MiB. */
	handle.Init.ChipSelectHighTime = QSPI_CS_HIGH_TIME_6_CYCLE;
	handle.Init.ClockMode = QSPI_CLOCK_MODE_0;
	handle.Init.FlashID = QSPI_FLASH_ID_1;
	handle.Init.DualFlash = QSPI_DUALFLASH_DISABLE;
	if (HAL_QSPI_Init(&handle) != HAL_OK) {
		return -1;
	}

	/* Boot/programming leaves the chip in standard SPI mode. Reject any
	 * unsupported part/state instead of issuing vendor-specific writes. */
	command.InstructionMode = QSPI_INSTRUCTION_1_LINE;
	command.Instruction = 0x9f; /* JEDEC ID */
	command.AddressMode = QSPI_ADDRESS_NONE;
	command.AlternateByteMode = QSPI_ALTERNATE_BYTES_NONE;
	command.DataMode = QSPI_DATA_1_LINE;
	command.NbData = sizeof(id);
	command.DdrMode = QSPI_DDR_MODE_DISABLE;
	command.DdrHoldHalfCycle = QSPI_DDR_HHC_ANALOG_DELAY;
	command.SIOOMode = QSPI_SIOO_INST_EVERY_CMD;
	if (HAL_QSPI_Command(&handle, &command, 100) != HAL_OK
			|| HAL_QSPI_Receive(&handle, id, 100) != HAL_OK) {
		return -1;
	}
	printf("ncnn_qspi: JEDEC=%02x%02x%02x\n", id[0], id[1], id[2]);
	if (!((id[0] == 0xef && id[1] == 0x40 && id[2] == 0x18)
			|| (id[0] == 0x20 && id[1] == 0xba && id[2] == 0x18))) {
		return -1;
	}

	/* Single-line memory-mapped read works without changing QE, dummy-cycle
	 * registers or nonvolatile status bits. No erase/write commands here. */
	command.Instruction = 0x03;
	command.AddressMode = QSPI_ADDRESS_1_LINE;
	command.AddressSize = QSPI_ADDRESS_24_BITS;
	command.NbData = 0;
	mapping.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
	return HAL_QSPI_MemoryMapped(&handle, &command, &mapping) == HAL_OK ? 0 : -1;
}
