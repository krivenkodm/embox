/* Read-only model storage for the STM32F746G-DISCO NCNN smoke test. */
#include <stdio.h>

#include "stm32746g_discovery_qspi.h"

static int read_register(QSPI_HandleTypeDef *handle, uint8_t instruction, uint8_t *value) {
	QSPI_CommandTypeDef command = {0};
	command.InstructionMode = QSPI_INSTRUCTION_1_LINE;
	command.Instruction = instruction;
	command.DataMode = QSPI_DATA_1_LINE;
	command.NbData = 1;
	command.DdrHoldHalfCycle = QSPI_DDR_HHC_ANALOG_DELAY;
	return HAL_QSPI_Command(handle, &command, 100) == HAL_OK
			&& HAL_QSPI_Receive(handle, value, 100) == HAL_OK ? 0 : -1;
}

int ncnn_stm32f746_qspi_map_lines(unsigned int data_lines) {
	QSPI_HandleTypeDef handle = {0};
	QSPI_CommandTypeDef command = {0};
	QSPI_MemoryMappedTypeDef mapping = {0};
	uint8_t id[3];
	uint8_t sr1, sr2;

	if (data_lines != 1 && data_lines != 4) {
		return -1;
	}
	/* QSPI holds read-only data, never code or dirty cache lines. Invalidate
	 * the whole mapped region so a preceding mode cannot mask a bad read.
	 * Both modes pay the same cache-maintenance cost, inside the QSPI timer. */
	SCB_InvalidateDCache_by_Addr((uint32_t *)0x90000000u, 0x01000000);

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

	if (id[0] == 0xef) {
		if (read_register(&handle, 0x05, &sr1) != 0
				|| read_register(&handle, 0x35, &sr2) != 0 || (sr1 & 1)) {
			return -1;
		}
		printf("ncnn_qspi: Winbond SR1=%02x SR2=%02x\n", sr1, sr2);
		if (data_lines == 4 && !(sr2 & 2)) {
			printf("ncnn_qspi: FAIL quad requires existing QE=1; status unchanged\n");
			return -1;
		}
	} else if (data_lines == 4) {
		printf("ncnn_qspi: FAIL quad supported only for Winbond ef4018\n");
		return -1;
	}

	/* Winbond FV/JV datasheet 8.2.9: 6Bh uses a one-line instruction and
	 * 24-bit address, eight dummy clocks, then four data lines. QE must
	 * already be set. Never issue Write Enable or a status/program/erase
	 * instruction; single-line mode remains available with QE=0. */
	command.Instruction = data_lines == 4 ? 0x6b : 0x03;
	command.DataMode = data_lines == 4 ? QSPI_DATA_4_LINES : QSPI_DATA_1_LINE;
	command.DummyCycles = data_lines == 4 ? 8 : 0;
	command.AddressMode = QSPI_ADDRESS_1_LINE;
	command.AddressSize = QSPI_ADDRESS_24_BITS;
	command.NbData = 0;
	mapping.TimeOutActivation = QSPI_TIMEOUT_COUNTER_DISABLE;
	if (HAL_QSPI_MemoryMapped(&handle, &command, &mapping) != HAL_OK) {
		return -1;
	}
	printf("ncnn_qspi: data_lines=%u opcode=0x%02lx dummy=%lu clock_hz=27000000\n",
			data_lines, (unsigned long)command.Instruction, (unsigned long)command.DummyCycles);
	return 0;
}

int ncnn_stm32f746_qspi_map(void) {
	return ncnn_stm32f746_qspi_map_lines(1);
}
