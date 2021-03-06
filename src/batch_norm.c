#include "uwnet.h"

#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include <float.h>
//#define FLT_EPSILON .00001f

matrix mean(matrix x, int spatial)
{
    matrix m = make_matrix(1, x.cols/spatial);
    int i, j;
    for(i = 0; i < x.rows; ++i){
        for(j = 0; j < x.cols; ++j){
            m.data[j/spatial] += x.data[i*x.cols + j];
        }
    }
    for(i = 0; i < m.cols; ++i){
        m.data[i] = m.data[i] / x.rows / spatial;
    }
    return m;
}

matrix variance(matrix x, matrix m, int spatial)
{
    matrix v = make_matrix(1, x.cols/spatial);
    // TODO: 7.1 - calculate variance
    for (int i = 0; i < x.rows; i++) {
      for (int j = 0; j < x.cols; j++) {
        v.data[j/spatial] += pow(x.data[i*x.cols + j] - m.data[j/spatial], 2.0);
      }
    }
    for (int i = 0; i < v.cols; i++) {
      v.data[i] = v.data[i] / x.rows / spatial;
    }
    return v;
}

matrix normalize(matrix x, matrix m, matrix v, int spatial)
{
    matrix norm = make_matrix(x.rows, x.cols);
    // TODO: 7.2 - normalize array, norm = (x - mean) / sqrt(variance + eps)
    for (int r = 0; r < x.rows; r++) {
      for (int c = 0; c < x.cols; c++) {
        int index = r * x.cols + c;
        //fprintf(stderr, "nor base: %f\n", sqrt(v.data[c/spatial] + FLT_EPSILON));
        norm.data[index] = (x.data[index] - m.data[c/spatial]) / sqrtf(v.data[c/spatial] + FLT_EPSILON);
      }
    }
    return norm;
}

matrix batch_normalize_forward(layer l, matrix x)
{
    float s = .1;
    int spatial = x.cols / l.rolling_mean.cols;
    if (x.rows == 1){
        return normalize(x, l.rolling_mean, l.rolling_variance, spatial);
    }
    matrix m = mean(x, spatial);
    //fprintf(stderr, "after mean\n");
    matrix v = variance(x, m, spatial);
    //fprintf(stderr, "after variance\n");
    matrix x_norm = normalize(x, m, v, spatial);
    //fprintf(stderr, "after normalize\n");
    scal_matrix(1-s, l.rolling_mean);
    axpy_matrix(s, m, l.rolling_mean);

    scal_matrix(1-s, l.rolling_variance);
    axpy_matrix(s, v, l.rolling_variance);

    free_matrix(m);
    free_matrix(v);

    free_matrix(l.x[0]);
    l.x[0] = x;

    return x_norm;
}


matrix delta_mean(matrix d, matrix variance, int spatial)
{
    matrix dm = make_matrix(1, variance.cols);
    // TODO: 7.3 - calculate dL/dmean
    for (int r = 0; r < d.rows; r++) {
      for (int c = 0; c < d.cols; c++) {
        // leave the second part as TODO
        dm.data[c/spatial] += (-d.data[r * d.cols + c] / sqrtf(variance.data[c/spatial] + FLT_EPSILON));
      }
    }
    return dm;
}

matrix delta_variance(matrix d, matrix x, matrix mean, matrix variance, int spatial)
{
    matrix dv = make_matrix(1, variance.cols);
    // TODO: 7.4 - calculate dL/dvariance
    assert(d.rows == x.rows);
    assert(d.cols == x.cols);
    for (int r = 0; r < d.rows; r++) {
      for (int c = 0; c < d.cols; c++) {
        int index = r * d.cols + c;
        float v_eps =  variance.data[c/spatial] + FLT_EPSILON;
        dv.data[c/spatial] += (-d.data[index] * (x.data[index] - mean.data[c/spatial]) / (v_eps * sqrtf(v_eps) * 2.0f));
      }
    }
    return dv;
}

matrix delta_batch_norm(matrix d, matrix dm, matrix dv, matrix mean, matrix variance, matrix x, int spatial)
{
    int i, j;
    matrix dx = make_matrix(d.rows, d.cols);
    // TODO: 7.5 - calculate dL/dx
    int m = d.rows * spatial;
    for (i = 0; i < dx.rows; i++) {
      for (j = 0; j < dx.cols; j++) {
        int index = i * dx.cols + j;
        int m_v_index = j / spatial;
        float v_eps = variance.data[j / spatial] + FLT_EPSILON;
        dx.data[index] = d.data[index] / sqrtf(v_eps)
                         +  2 * dv.data[m_v_index] * (x.data[index] - mean.data[m_v_index]) / m
                         + dm.data[m_v_index] / m;
      }
    }
    return dx;
}

matrix batch_normalize_backward(layer l, matrix d)
{
    int spatial = d.cols / l.rolling_mean.cols;
    matrix x = l.x[0];

    matrix m = mean(x, spatial);
    matrix v = variance(x, m, spatial);

    matrix dm = delta_mean(d, v, spatial);
    matrix dv = delta_variance(d, x, m, v, spatial);
    matrix dx = delta_batch_norm(d, dm, dv, m, v, x, spatial);

    free_matrix(m);
    free_matrix(v);
    free_matrix(dm);
    free_matrix(dv);

    return dx;
}
