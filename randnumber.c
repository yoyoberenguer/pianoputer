
// C program for generating a 
// random number in a given range 
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// RETURN A RANDOMIZE FLOAT NUMBER BETWEEN LOWER AND UPPER LIMITS
float randRangeFloat(float lower, float upper){
    return lower + ((float)rand()/(float)(RAND_MAX)) * (upper - lower);
}

// RETURN A RANDOMIZE INTEGER VALUE BETWEEN LOWER AND UPPER LIMITS
int randRange(int lower, int upper)
{
    return (rand() % (upper - lower  + 1)) + lower;
}

//
//int main(){
//return 0;
//}