// RM3100 Magnetometer Calibration Parameters
// Generated on 20250529_225522

#ifndef RM3100_CALIBRATION_H
#define RM3100_CALIBRATION_H

// Hard iron offset (µT)
static const float rm3100_hard_iron_offset[3] = {
    15.657647f,  // X offset
    10.993677f,  // Y offset
    52.374377f   // Z offset
};

// Soft iron correction matrix
static const float rm3100_soft_iron_matrix[3][3] = {
    {0.040688f, -0.000244f, -0.002581f},
    {-0.000244f, 0.038607f, 0.001081f},
    {-0.002581f, 0.001081f, 0.044625f}
};

// Calibration function
// Usage: apply_calibration(raw_x, raw_y, raw_z, &cal_x, &cal_y, &cal_z)
static inline void rm3100_apply_calibration(float raw_x, float raw_y, float raw_z,
                                           float* cal_x, float* cal_y, float* cal_z) {
    // Apply hard iron correction
    float corrected[3] = {
        raw_x - rm3100_hard_iron_offset[0],
        raw_y - rm3100_hard_iron_offset[1],
        raw_z - rm3100_hard_iron_offset[2]
    };

    // Apply soft iron correction matrix
    *cal_x = rm3100_soft_iron_matrix[0][0] * corrected[0] +
             rm3100_soft_iron_matrix[0][1] * corrected[1] +
             rm3100_soft_iron_matrix[0][2] * corrected[2];
    *cal_y = rm3100_soft_iron_matrix[1][0] * corrected[0] +
             rm3100_soft_iron_matrix[1][1] * corrected[1] +
             rm3100_soft_iron_matrix[1][2] * corrected[2];
    *cal_z = rm3100_soft_iron_matrix[2][0] * corrected[0] +
             rm3100_soft_iron_matrix[2][1] * corrected[1] +
             rm3100_soft_iron_matrix[2][2] * corrected[2];
}

#endif // RM3100_CALIBRATION_H
