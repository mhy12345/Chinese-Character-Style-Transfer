#include <stdio.h>
#include <memory.h>
#define N 64


float flood_fill(const float* real, const float* fake) {
    static const int mov[4][2] = {{0,-1}, {-1,0}, {0,1}, {1,0}};
    static int dis[N][N];
    static int q[N*N][2];
    int head = -1, tail = -1;
    memset(dis,-1,sizeof(dis));
    for (int i=0;i<N;i++)
        for (int j=0;j<N;j++)
            if (real[i*N+j] < -.8) {//Black pixel 
                dis[i][j] = 0;
                ++tail;
                q[tail][0] = i;
                q[tail][1] = j;
            }
    while (head < tail) {
        ++head;
        int sx,sy;
        int x,y;
        sx = q[head][0];
        sy = q[head][1];
        for (int i=0;i<4;i++) {
            x = sx + mov[i][0];
            y = sy + mov[i][1];
            if (x<N && y<N && x>=0 && y>=0 && dis[x][y] == -1) {
                dis[x][y] = dis[sx][sy] + 1;
                ++tail;
                q[tail][0]=x;
                q[tail][1]=y;
            }
        }
    }
    int sum = 0, tot = 0;
    for (int i=0;i<N;i++) {
        for (int j=0;j<N;j++) {
            if (fake[i*N+j] < -.8) {
                sum += dis[i][j];
                tot ++;
            }
        }
    }
    //printf("HAHA\n");
    return tot?1.0*sum/tot:8;
}
