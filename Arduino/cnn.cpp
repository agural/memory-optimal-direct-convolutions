/************************************************************************
 * Title:      AVR CNN                                                  *
 * Desciption: Code for running small CNNs using only 2K on-board SRAM. *
 * Author:     *Anonymous*                                              *
 * Date:       2018-11-21                                               *
 * Software:   Arduino                                                  *
 * Target:     ATmega328 (or similar device with 2K SRAM)               *
 ************************************************************************/

/********** INCLUDES **********/
#include <avr/io.h>
//#include <stdlib.h>
//#include <stdarg.h>


/********** CNN-RELEVANT DEFINES **********/
// The main data types.
typedef unsigned char uc;
typedef unsigned int  ui;

// Generally useful defines.
#define min(a,b) ((a)<(b)?(a):(b))
#define max(a,b) ((a)>(b)?(a):(b))
#define abs(x) ((x)>0?(x):-(x))

// https://www.avrfreaks.net/comment/617981#comment-617981
#define AIL(x)   static x __attribute__ ((always_inline)); static x

// Allocate SRAM space for CNN weights/activations.
#define SIZE                1960

// All of the different types of layers supported.
#define LOAD_IMAGE_AVERAGE2 0
#define CONV_3x3_RELU       1
#define DENSE_OUT           2
#define MAX_POOL2           3


/********** SERIAL COMMUNICATIONS **********/
// https://gist.github.com/rms95/5887356
// https://www.avrfreaks.net/forum/uart-code-attiny2313-not-working-atmega328
#define UCSRA   UCSR0A
#define UCSRB   UCSR0B
#define UCSRC   UCSR0C
#define UBRRH   UBRR0H
#define UBRRL   UBRR0L
#define UDRE    UDRE0
#define UDR     UDR0
#define RXC     RXC0
#define RXEN    RXEN0
#define TXEN    TXEN0
#define UCSZ1   UCSZ01
#define UCSZ0   UCSZ00 
#define U2X     U2X0
#define URSEL   7

#define SERIAL_receiveReady (!!(UCSRA&(1<<RXC)))
#define SERIAL_sendReady    (!!(UCSRA&(1<<UDRE)))
#define SERIAL_byte UDR

void SERIAL_init(uint32_t baudrate) {
	baudrate = ((F_CPU/(baudrate*8UL))-1);
	UCSRA|=(1<<U2X);
	UCSRB=(1<<RXEN)|(1<<TXEN);
	UCSRC=(1<<URSEL)|(1<<UCSZ0)|(1<<UCSZ1);
	UBRRH=(baudrate >> 8) ;
	UBRRL=baudrate;
}


/********** CNN Utilities **********/
// a is the portion of SRAM pre-allocated for CNN-relevant data.
// It is split into four sections:
//   1. CNN specification (layers, sizes, quantization scale).
//   2. Weights and Biases for each layer.
//   3. Current layer activations.
//   4. Scratch area for computing next activations.
volatile uc a[SIZE];

// Start of activation section of a (see section (3.) above).
// Initially gives the byte address; later gives the nibble address.
ui s = 0;

// Get an unsigned nibble of a (for activations).
inline uc ag(ui nibble_ptr) {
	uc part = a[nibble_ptr >> 1];
	if (nibble_ptr & 1) return (part & 0x0F);
	else return (part >> 4);
}

// Get a signed nibble of a (for weights).
inline char aw(ui nibble_ptr) {
	uc part = a[nibble_ptr >> 1];
	char result = part >> 4;
	if (nibble_ptr & 1) result = part & 0x0F;
	if (result & 0x08)  result |= 0xF0;
	return result;
}

// Get a signed byte of a (for biases).
inline char ab(ui nibble_ptr) {
	char result = ag(nibble_ptr+1);
	result |= (ag(nibble_ptr) << 4);
	return result;
}

// Set a nibble of a (for activations).
inline void as(ui nibble_ptr, uc value) {
	ui ptr = nibble_ptr >> 1;
	uc part = a[ptr];
	if(nibble_ptr & 1) a[ptr] = (part & 0xF0) | (value & 0x0F);
	else a[ptr] = (part & 0x0F) | (value << 4);
}


/********** DEBUGGING HELPERS **********/
// https://www.avrfreaks.net/forum/soft-c-avrgcc-monitoring-stack-usage
// http://blog.qartis.com/avr-initn-sections-and-gcc-flto/
// https://stackoverflow.com/questions/20059673/print-out-value-of-stack-pointer
// https://www.avrfreaks.net/forum/memory-barrier-what-it-does-and-what-it-does-not-do
#define STACK_CANARY 0xaa
extern uint8_t __bss_end;
extern uint8_t __stack;

// Paints the region from .data to .stack so that stack usage can be determined later.
// This runs before the start of main().
__attribute__ ((section(".init1"), naked, used))
void stack_paint(void) {
#if 0
    for(uc *p = &__bss_end; p <= &__stack; ++p) *p = STACK_CANARY;
#else
    __asm volatile ("    ldi r30,lo8(__bss_end)\n"
                    "    ldi r31,hi8(__bss_end)\n"
                    "    ldi r24,lo8(0xaa)\n" /* STACK_CANARY */
                    "    ldi r25,hi8(__stack)\n"
                    "    rjmp .cmp\n"
                    ".loop:\n"
                    "    st Z+,r24\n"
                    ".cmp:\n"
                    "    cpi r30,lo8(__stack)\n"
                    "    cpc r31,r25\n"
                    "    brlo .loop\n"
                    "    breq .loop"::);
#endif
}

// Manually paint the stack area during program execution.
// Set depth small enough to not interfere with the current stack pointer.
void stack_paint_force(uc depth) {
	for(uc *p = &__bss_end; p <= (&__bss_end) + depth; ++p) *p = STACK_CANARY;
}

// Finds the stack pointer location and prints to serial out.
//AIL(void check_sp(uc msg));
//__attribute__ ((noinline))
void check_sp(uc msg) {
	uc p = 0;
	__asm volatile("": : :"memory");
	uc pdist = (uc)(&__stack - &p);
	uc sdist = 0;
    for(const uc *p = &__bss_end; *p == STACK_CANARY && p <= &__stack; p++) sdist++;
	while(!SERIAL_sendReady);
	SERIAL_byte = msg;
	while(!SERIAL_sendReady);
	SERIAL_byte = sdist;
	//while(!SERIAL_sendReady);
	//SERIAL_byte = *((uc*)(29));
	//while(!SERIAL_sendReady);
	//SERIAL_byte = *((uc*)(28));
	while(!SERIAL_sendReady);
	SERIAL_byte = pdist;
}


/********** CNN Layer LOAD_IMAGE_AVERAGE2 **********/
// Load the image with pre-average-pooling 2x2.
// The average pooling allows the image to fit in the activations section of a.
// This function also converts the byte pointer s to a nibble pointer.
//AIL(void load_image_average2(uc &dim0, uc &dim1, uc &dim2));
inline void load_image_average2(uc &dim0, uc &dim1, uc &dim2) {
	uc dim12 = dim1 * dim2;
	ui ptr = s;
	for(uc r = 0; r < dim0; ++r) {
		for(uc c = 0; c < dim1; ++c) {
			for(uc f = 0; f < dim2; ++f) {
				while(!SERIAL_receiveReady);
				uc b = SERIAL_byte;
				a[ptr] = b;
				ptr += 1;
				if(r & c & 1) {
					int accum = a[ptr-1] + a[ptr-dim2-1] + a[ptr-dim12+(c-1)*dim2-1] + a[ptr-dim12+(c-2)*dim2-1];
					a[ptr-dim12+(c>>1)*dim2-dim2-1] = (accum >> 6); // + ((accum >> 5) & 1); // un-comment for mid-tread.
					if(f == dim2-1) {
						ptr -= 2*dim2;
						if(c == dim1-1) {
							ptr -= (dim12 >> 1);
						}
					}
				}
			}
		}
	}

	// Compress to 4b.
	ptr = s;
	int ptr2 = s;
	for(int i = 0; i < dim0 * (dim12 >> 1); i += 2) {
		a[ptr] = a[ptr2] << 4;
		a[ptr] += a[ptr2+1];
		ptr += 1;
		ptr2 += 2;
	}
	s <<= 1; // Convert to a nibble pointer.
	dim0  >>= 1;
	dim1  >>= 1;
}


/********** CNN Layer MAX_POOL2 **********/
// Implements the max pooling 2x2 layer.
inline void max_pool2(uc &dim0, uc &dim1, uc &dim2) {
	uc dim12 = dim1 * dim2;
	ui nib_ptr = s;
	ui loc;
	for(uc r = 0; r < dim0; r+=2) {
		for(uc c = 0; c < dim1; c+=2) {
			for(uc f = 0; f < dim2; f++) {
				loc = s + r*dim12 + c*dim2 + f;
				uc value = ag(loc);
				value = max(value, ag(loc+dim2));
				value = max(value, ag(loc+dim12));
				value = max(value, ag(loc+dim12+dim2));
				as(nib_ptr, value);
				nib_ptr += 1;
			}
		}
	}
	dim0 >>= 1;
	dim1 >>= 1;
}


/********** CNN Layer DENSE_OUT **********/
// Implements the final fully connected layer of a CNN.
// Outputs 16b logits starting at pointer s.
// Also returns the index of the max logit (ie the class).
// Important: currently requires an input dimension dim0 < 256.
inline uc dense_out(uc &dim0, ui &weight_ptr, uc &config_ptr) {
	uc dim1 = ag(config_ptr + 1);

	// Move activations to end of work area.
	ui a_ptr = s + dim0 - 1;
	ui b_ptr = 2*SIZE - 1;
	while(a_ptr >= s) {
		as(b_ptr, ag(a_ptr));
		a_ptr -= 1;
		b_ptr -= 1;
	}
	a_ptr += 1;
	b_ptr += 1;

	// Matrix-vector product.
	int *aout = (int*)(&(a[s/2]));
	for(uc fo = 0; fo < dim1; ++fo) aout[fo] = 0;
	for(uc fi = 0; fi < dim0; ++fi) {
		uc xi = ag(b_ptr);
		b_ptr += 1;
		for(uc fo = 0; fo < dim1; ++fo) {
			aout[fo] += xi * aw(weight_ptr);
			weight_ptr += 1;
		}
	}
	uc sL = ag(config_ptr + 2);
	uc best = 0;
	for(uc fo = 0; fo < dim1; ++fo) {
		aout[fo] <<= sL;
		aout[fo] += ab(weight_ptr);
		weight_ptr += 2;
		if(aout[fo] > aout[best]) best = fo;
	}
	dim0 = dim1;
	return best;
}


/********** CNN Layer CONV_3x3_RELU **********/
// This is a special quantization-like function for reducing precision.
// It mimics numpy/tf's rounding which rounds X.5 to the nearest even number.
inline void shift_right(int &a, uc &s) {
	uc mid = (1 << s);                    // = 1/2 after converting to bitwidth.
	uc frac = a & ((mid << 1) - 1);       // The fractional part after converting to bitwidth.
	a >>= s + 1;                          // Convert to appropriate bitwidth.
	a += (frac >= mid);                   // Round based on 1/2's place.
	a -= ((frac == mid) && (a & 1));      // Round 0.5 to nearest even.
	if(a > 15) a = 15;                    // Clipping.
}

// Computes the top row convolution.
//AIL(void conv_3x3_relu_row(ui &in_ptr, ui &out_ptr, uc &len, \
	ui &weight_ptr, ui &bias_ptr, uc &dim0, uc &dim1, uc &dim2, uc &dim3, \
	uc &sL, uc &sR));
inline void conv_3x3_relu_row(ui &in_ptr, ui &out_ptr, uc &len,
		ui &weight_ptr, ui &bias_ptr, uc &dim0, uc &dim1, uc &dim2, uc &dim3,
		uc &sL, uc &sR) {
	uc dim12 = dim1 * dim2;
	for(uc i = 0; i < len; ++i) {
		for(uc fo = 0; fo < dim3; ++fo) {
			int accum = 0;
			for(uc r = 0; r < 3; ++r) {
				for(uc c = 0; c < 3; ++c) {
					for(uc fi = 0; fi < dim2; ++fi) {
						accum += aw(weight_ptr) * ag(in_ptr);
						weight_ptr += 1;
						in_ptr += 1;
					}
				}
				in_ptr += dim12 - 3*dim2;
			}
			accum <<= sL;
			accum += ab(bias_ptr);
			bias_ptr += 2;
			if(accum < 0) accum = 0;
			shift_right(accum, sR);
			as(out_ptr, accum);
			out_ptr += 1;
			in_ptr  -= 3*dim12;
		}
		weight_ptr -= 3*3*dim2*dim3;
		bias_ptr -= 2*dim3;
		in_ptr += dim2;
	}
	in_ptr += 2*dim2;
	dim0 -= 1;
}

// Computes the top row convolution, but with a transposed weight kernel.
// This allows for convolution with a transposed input image.
//AIL(void conv_3x3_relu_rowT(ui &in_ptr, ui &out_ptr, uc &len, \
	ui &weight_ptr, ui &bias_ptr, uc &dim0, uc &dim1, uc &dim2, uc &dim3, \
	uc &sL, uc &sR));
inline void conv_3x3_relu_rowT(ui &in_ptr, ui &out_ptr, uc &len,
		ui &weight_ptr, ui &bias_ptr, uc &dim0, uc &dim1, uc &dim2, uc &dim3,
		uc &sL, uc &sR) {
	uc dim12 = dim1 * dim2;
	for(uc i = 0; i < len; ++i) {
		for(uc fo = 0; fo < dim3; ++fo) {
			int accum = 0;
			for(uc r = 0; r < 3; ++r) {
				for(uc c = 0; c < 3; ++c) {
					for(uc fi = 0; fi < dim2; ++fi) {
						accum += aw(weight_ptr) * ag(in_ptr);
						weight_ptr += 1;
						in_ptr += 1;
					}
					weight_ptr += 2*dim2;
				}
				weight_ptr -= 8*dim2;
				in_ptr += dim12 - 3*dim2;
			}
			weight_ptr += 6*dim2;
			accum <<= sL;
			accum += ab(bias_ptr);
			bias_ptr += 2;
			if(accum < 0) accum = 0;
			shift_right(accum, sR);
			as(out_ptr, accum);
			out_ptr += 1;
			in_ptr  -= 3*dim12;
		}
		weight_ptr -= 3*3*dim2*dim3;
		bias_ptr -= 2*dim3;
		in_ptr += dim2;
	}
	in_ptr += 2*dim2;
	dim0 -= 1;
}

// Computes the transpose of the first two dimensions of the
// dim0 x dim1 x dim2 input image pointed to by in_ptr.
// It computes the transpose in-place using O(1) additional memory and is based on
// this solution:
//   https://softwareengineering.stackexchange.com/questions/271713/transpose-a-matrix-without-a-buffering-one
AIL(void transpose_fast(ui in_ptr, uc &dim0, uc &dim1, uc &dim2));
inline void transpose_fast(ui in_ptr, uc &dim0, uc &dim1, uc &dim2) {
	uc k;//, first;
	ui next, temp;
	ui start = (ui)(dim0) * dim1 - 2;
	for(; start; --start) {
		// Check if valid cycle start.
		next = start;
		temp = -1;
		do {
			temp += 1;
			next = (next % dim0) * dim1 + next / dim0;
		} while(next > start);

		// Rotate elements in the cycle.
		if(next >= start && temp) {
			next = start;
			for(k = 0; k < dim2; ++k) as(in_ptr-k-1, ag(in_ptr + start * dim2 + k));
			do {
				temp = (next % dim0) * dim1 + next / dim0;
				for(k = 0; k < dim2; ++k)
					as(in_ptr + next * dim2 + k, (temp == start) ? ag(in_ptr-k-1) : ag(in_ptr + temp * dim2 + k));
				next = temp;
			} while(next > start);
		}
	}
	dim0 = dim1 ^ dim0;
	dim1 = dim1 ^ dim0;
	dim0 = dim1 ^ dim0;
}

inline void transpose(ui in_ptr, uc &dim0, uc &dim1, uc &dim2) {
	uc k, first;
	ui next, temp;
	ui start = (ui)(dim0) * dim1 - 2;
	for(; start; --start) {
		// Check if valid cycle start.
		next = start;
		temp = -1;
		do {
			temp += 1;
			next = (next % dim0) * dim1 + next / dim0;
		} while(next > start);

		// Rotate elements in the cycle.
		if(next >= start && temp) {
			for(k = 0; k < dim2; ++k) {
				next = start;
				first = ag(in_ptr + start * dim2 + k);
				do {
					temp = (next % dim0) * dim1 + next / dim0;
					as(in_ptr + next * dim2 + k, (temp == start) ? first : ag(in_ptr + temp * dim2 + k));
					next = temp;
				} while(next > start);
			}
		}
	}
	dim0 = dim1 ^ dim0;
	dim1 = dim1 ^ dim0;
	dim0 = dim1 ^ dim0;
}


// Computes the successor function for the herringbone shape.
//AIL(ui herringbone_successor(uc n, ui x));
inline ui herringbone_successor(uc n, ui x) {
	uc i = x / n;
	uc j = x % n;
	if(j >= i) return (ui)(i) * ((n<<1) - i - 1) + j;
	else       return (ui)(j) * ((n<<1) - j - 2) + n + i - 1;
}

// Undoes the herringbone pattern. dim1 must equal dim0.
//AIL(void herringbone(ui in_ptr, uc dim0, uc dim2));
inline void herringbone(ui in_ptr, uc dim0, uc dim2) {
	uc k, first;
	ui next, temp;
	ui start = (ui)(dim0) * dim0 - 2;
	for(; start; --start) {
		// Check if valid cycle start.
		next = start;
		temp = -1;
		do {
			temp += 1;
			next = herringbone_successor(dim0, next);
		} while(next > start);

		// Rotate elements in the cycle.
		if(next >= start && temp) {
			for(k = 0; k < dim2; ++k) {
				next = start;
				first = ag(in_ptr + start * dim2 + k);
				do {
					temp = herringbone_successor(dim0, next);
					as(in_ptr + next * dim2 + k, (temp == start) ? first : ag(in_ptr + temp * dim2 + k));
					next = temp;
				} while(next > start);
			}
		}
	}
}

// Top-level function for the 3x3 convolution layer.
//AIL(void conv_3x3_relu(uc &dim0, uc &dim1, uc &dim2, ui &weight_ptr, uc config_ptr));
inline void conv_3x3_relu(uc &dim0, uc &dim1, uc &dim2, ui &weight_ptr, uc config_ptr) {
	uc dim3 = ag(config_ptr + 1);
	ui bias_ptr = weight_ptr + 3 * 3 * dim2 * dim3;
	uc dim12 = dim1 * dim2;
	uc sL = ag(config_ptr + 2);
	uc sR = ag(config_ptr + 3);

	bool use_herringbone = (ui)(dim0-2)*dim1*dim3 + 3*dim2 > (2*SIZE - s);

	// Move activations to end of work area.
	ui a_ptr = s + dim0 * dim12 - 1;
	ui b_ptr = 2*SIZE - 1;
	while(a_ptr >= s) {
		as(b_ptr, ag(a_ptr));
		a_ptr -= 1;
		b_ptr -= 1;
	}
	a_ptr += 1;
	b_ptr += 1;

	// Perform the convolution.
	uc dim0x = dim0;
	uc dim1x = dim1;
	uc diff = dim1x - dim0x;
	//* Row-by-row convolution (not space efficient).
	while(!use_herringbone && dim0x > 2) {
		uc len = dim1x - 2;
		conv_3x3_relu_row(b_ptr, a_ptr, len,
			weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
	}
	/* Col-by-col convolution (not space efficient).
	while(dim1x > 2) {
		transpose(b_ptr, dim0x, dim1x, dim2);
		uc len = dim1x - 2;
		conv_3x3_relu_rowT(b_ptr, a_ptr, len,
			weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
		transpose(b_ptr, dim0x, dim1x, dim2);
	}*/
	//* Herringbone: alternating row-col (space efficient; computationally intensive).
	//check_sp(0x20);
	if(use_herringbone) {
		if(diff > 0) {
			transpose_fast(b_ptr, dim0x, dim1x, dim2);
			uc len = dim1x - 2;
			for(uc i = 0; i < diff; ++i) {
				conv_3x3_relu_rowT(b_ptr, a_ptr, len,
					weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
			}
			transpose_fast(b_ptr, dim0x, dim1x, dim2);
		} else {
			uc len = dim1x - 2;
			for(uc i = 0; i < 1-diff; ++i) {
				//check_sp(0x21);
				conv_3x3_relu_row(b_ptr, a_ptr, len,
					weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
				__asm volatile("": : :"memory");
				//check_sp(0x22);
			}
		}
		//check_sp(0x23);
		while(dim0x > 2) {
			//check_sp(0x30);
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 0xe0;
			transpose_fast(b_ptr, dim0x, dim1x, dim2);
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 0xe1;
			//check_sp(0x31);
			uc len = dim1x - 2;
			conv_3x3_relu_rowT(b_ptr, a_ptr, len,
				weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 0xe2;
			//check_sp(0x32);
			transpose_fast(b_ptr, dim0x, dim1x, dim2);
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 0xe3;
			//check_sp(0x33);
			conv_3x3_relu_row(b_ptr, a_ptr, len,
				weight_ptr, bias_ptr, dim0x, dim1x, dim2, dim3, sL, sR);
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 0xe4;
			//check_sp(0x34);
		}
	}
	weight_ptr = bias_ptr + 2*dim3;
	dim0 -= 2;
	dim1 -= 2;
	dim2 = dim3;

	// Undo the herringbone pattern produced by the space-efficient convolution.
	if(use_herringbone) {
		ui hb_ptr = s + (int)(abs(diff)) * min(dim0, dim1) * dim2;
		herringbone(hb_ptr, dim0, dim2);
		if(diff > 0) {
			// TODO: transpose overwrites a few elements before the start pointer.
			// This will corrupt the data before it.
			transpose(hb_ptr, dim0, dim0, dim2);
			transpose(s, dim0, dim1, dim2);
		}
	}
}

/********** MAIN CODE **********/
// Sets up serial communications and loads the CNN specification.
__attribute__ ((noinline))
void setup() {
	DDRB = 0b00000000; // All inputs.
	DDRC = 0b00000000; // All inputs.
	DDRD = 0b00000010; // RX on PD0; TX on PD1.
	uint32_t baud = 115200UL;
	SERIAL_init(baud);
	while(!SERIAL_receiveReady);
	uc b = SERIAL_byte;
	s += b;
	s <<= 8;
	while(!SERIAL_receiveReady);
	b = SERIAL_byte;
	s += b;
	for(ui i = 0; i < s; ++i) {
		while(!SERIAL_receiveReady);
		uc b = SERIAL_byte;
		a[i] = b;
	}
}

// The main program does the following:
//   1. Load the NN specification.
//   2. Loop infinitely:
//      3. Wait for an input image.
//      4. Compute CNN inference for the input image.
//      5. Outputs the predicted image class.
__attribute__ ((noreturn))
void main() {
	setup();
	//check_sp(0x00);
	for(;;) {
		// Number of layers and input dims are specified at the start of a.
		uc num_layers = ag(0);
		uc dim0 = a[1];
		uc dim1 = a[2];
		uc dim2 = ag(1);

		// Find the weights/biases address of a.
		ui weight_ptr = 6; // 6 is the start nibble of layer specifications.
		for(uc layer = 0; layer < num_layers; ++layer) {
			if(ag(weight_ptr) == LOAD_IMAGE_AVERAGE2) {
				weight_ptr += 1;
				continue;
			}
			if(ag(weight_ptr) == MAX_POOL2) {
				weight_ptr += 1;
				continue;
			}
			if(ag(weight_ptr) == CONV_3x3_RELU) {
				weight_ptr += 4;
				continue;
			}
			if(ag(weight_ptr) == DENSE_OUT) {
				weight_ptr += 4;
				continue;
			}
		}

		// Run forward inference through each layer.
		uc layer_ptr = 6; // 6 is the start nibble of layer specifications.
		uc result = 0;
		for(uc layer = 0; layer < num_layers; ++layer) {
			if(ag(layer_ptr) == LOAD_IMAGE_AVERAGE2) {
				load_image_average2(dim0, dim1, dim2);
				layer_ptr += 1;
			}
			else if(ag(layer_ptr) == MAX_POOL2) {
				max_pool2(dim0, dim1, dim2);
				layer_ptr += 1;
			}
			else if(ag(layer_ptr) == CONV_3x3_RELU) {
				conv_3x3_relu(dim0, dim1, dim2, weight_ptr, layer_ptr);
				layer_ptr += 4;
			}
			else if(ag(layer_ptr) == DENSE_OUT) {
				dim0 *= dim1 * dim2;
				result = dense_out(dim0, weight_ptr, layer_ptr);
				dim1 = 1;
				dim2 = 4; // This is just used to indicate 4 nibbles per output.
			}
			else {
				break;
			}
			//while(!SERIAL_sendReady);
			//SERIAL_byte = 16 + layer;
			//check_sp(0x10 + layer);
		}

		s >>= 1; // Prepare for the next iteration, where s is a byte pointer again.
		for(ui ptr = s; ptr < s + (ui)(dim0)*dim1*dim2/2; ++ptr) {
			while(!SERIAL_sendReady);
			SERIAL_byte = a[ptr];
		}

		while(!SERIAL_sendReady);
		SERIAL_byte = result;
	}
}

