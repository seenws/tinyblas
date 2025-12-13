#include <stdio.h>

int main(void)
{
    int a[4] = {0,1,2,3};

    for (int i = 0; i < 4; ++i)
        printf("%d-%p\n", a[i], &a[i]);

    printf("%zu\n", sizeof(int));

    return 0;
}
