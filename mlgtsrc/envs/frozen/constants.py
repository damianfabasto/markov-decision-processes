# Code from https://github.com/wesley-smith/CS7641-assignment-4/tree/f3d86e37504dda563f65b3267610a30f09d01c77
FL4x4 = '4x4'
FL8x8 = '8x8'
TERM_STATE_MAP = {
    FL4x4: [5, 7, 8, 12],
    #FL8x8: [10, 29, 30, 31, 45, 46, 50]
    #FL8x8: [8]
    FL8x8: [15, 19, 29, 35, 41, 42, 46, 49, 52, 54, 60]
}
GOAL_STATE_MAP = {
    FL4x4: [15],
    FL8x8: [63]
}