#ifndef OTHELLO_H
#define OTHELLO_H

#include <stdint.h>
#include <string.h>
#include <stdlib.h>

#define OTH_BOARD_SIZE 8
#define OTH_NUM_CELLS 64
#define OTH_NUM_ACTIONS 65  // 64 squares + 1 pass
#define OTH_OBS_DIM 192     // 3 planes of 64 floats

#define OTH_BLACK 0
#define OTH_WHITE 1

#define OTH_ACTION_PASS 64

// Starting positions:
// Black: d5 (bit 35) and e4 (bit 28)
// White: d4 (bit 27) and e5 (bit 36)
// bit i = row i/8, col i%8, row 0 = top, col 0 = left
#define OTH_START_BLACK 0x0000000810000000ULL
#define OTH_START_WHITE 0x0000001008000000ULL

static inline int popcount64(uint64_t x) {
    return __builtin_popcountll(x);
}

typedef struct {
    uint64_t black;
    uint64_t white;
    int current_player;  // OTH_BLACK or OTH_WHITE
    int pass_count;      // consecutive passes (2 = game over)
    int done;
    float reward;        // from perspective of the agent
    int episode_id;
    int move_count;
    int invalid_moves;
    int corner_captures;
} Othello;

// Direction shifts and masks to prevent wrapping
// Directions: N, NE, E, SE, S, SW, W, NW
static const int OTH_DIR_SHIFT[8] = { -8, -7, 1, 9, 8, 7, -1, -9 };

// Masks to prevent wrapping when shifting
// For eastward directions (NE, E, SE): mask out column 0 (leftmost after shift)
// For westward directions (NW, W, SW): mask out column 7 (rightmost after shift)
static const uint64_t OTH_NOT_A_FILE = 0xFEFEFEFEFEFEFEFEULL; // not column 0
static const uint64_t OTH_NOT_H_FILE = 0x7F7F7F7F7F7F7F7FULL; // not column 7

static inline uint64_t oth_shift(uint64_t board, int dir) {
    // Apply directional shift with wrapping prevention
    uint64_t result;
    switch (dir) {
        case 0: result = board >> 8; break;                                    // N
        case 1: result = (board >> 7) & OTH_NOT_A_FILE; break;                // NE
        case 2: result = (board << 1) & OTH_NOT_A_FILE; break;                // E
        case 3: result = (board << 9) & OTH_NOT_A_FILE; break;                // SE
        case 4: result = board << 8; break;                                    // S
        case 5: result = (board << 7) & OTH_NOT_H_FILE; break;                // SW
        case 6: result = (board >> 1) & OTH_NOT_H_FILE; break;                // W
        case 7: result = (board >> 9) & OTH_NOT_H_FILE; break;                // NW
        default: result = 0; break;
    }
    return result;
}

static inline uint64_t oth_get_moves(uint64_t my_board, uint64_t opp_board) {
    uint64_t empty = ~(my_board | opp_board);
    uint64_t moves = 0;

    for (int d = 0; d < 8; d++) {
        // Find opponent pieces adjacent to my pieces in this direction
        uint64_t candidates = oth_shift(my_board, d) & opp_board;

        // Follow the chain of opponent pieces
        while (candidates) {
            uint64_t next = oth_shift(candidates, d) & opp_board;
            // The end of the chain that lands on empty is a valid move
            uint64_t end = oth_shift(candidates, d) & empty;
            moves |= end;
            candidates = next;
        }
    }
    return moves;
}

static inline uint64_t oth_resolve_flips(uint64_t my_board, uint64_t opp_board, int sq) {
    uint64_t move_bit = 1ULL << sq;
    uint64_t flips = 0;

    for (int d = 0; d < 8; d++) {
        uint64_t line_flips = 0;
        uint64_t cursor = oth_shift(move_bit, d);

        while (cursor & opp_board) {
            line_flips |= cursor;
            cursor = oth_shift(cursor, d);
        }

        // If the chain ends on one of my pieces, the flips are valid
        if (cursor & my_board) {
            flips |= line_flips;
        }
    }
    return flips;
}

static inline void oth_reset(Othello *g) {
    g->black = OTH_START_BLACK;
    g->white = OTH_START_WHITE;
    g->current_player = OTH_BLACK;
    g->pass_count = 0;
    g->done = 0;
    g->reward = 0.0f;
    g->move_count = 0;
    g->invalid_moves = 0;
    g->corner_captures = 0;
}

static inline void oth_apply_move(Othello *g, int action) {
    uint64_t *my_board = (g->current_player == OTH_BLACK) ? &g->black : &g->white;
    uint64_t *opp_board = (g->current_player == OTH_BLACK) ? &g->white : &g->black;

    if (action == OTH_ACTION_PASS) {
        g->pass_count++;
    } else {
        uint64_t flips = oth_resolve_flips(*my_board, *opp_board, action);
        *my_board |= (1ULL << action) | flips;
        *opp_board &= ~flips;
        g->pass_count = 0;
        g->move_count++;

        // Track corner captures
        if (action == 0 || action == 7 || action == 56 || action == 63) {
            g->corner_captures++;
        }
    }

    // Switch player
    g->current_player ^= 1;
}

static inline int oth_check_terminal(Othello *g) {
    if (g->pass_count >= 2) {
        g->done = 1;
        int black_count = popcount64(g->black);
        int white_count = popcount64(g->white);
        // Reward from black's perspective: +1 win, -1 loss, 0 draw
        if (black_count > white_count) {
            g->reward = 1.0f;
        } else if (white_count > black_count) {
            g->reward = -1.0f;
        } else {
            g->reward = 0.0f;
        }
        return 1;
    }
    return 0;
}

static inline void oth_write_obs(Othello *g, float *obs, int agent_color) {
    uint64_t my_board = (agent_color == OTH_BLACK) ? g->black : g->white;
    uint64_t opp_board = (agent_color == OTH_BLACK) ? g->white : g->black;
    uint64_t legal = oth_get_moves(my_board, opp_board);

    // Plane 0: my pieces
    for (int i = 0; i < 64; i++) {
        obs[i] = (my_board >> i) & 1 ? 1.0f : 0.0f;
    }
    // Plane 1: opponent pieces
    for (int i = 0; i < 64; i++) {
        obs[64 + i] = (opp_board >> i) & 1 ? 1.0f : 0.0f;
    }
    // Plane 2: legal moves
    for (int i = 0; i < 64; i++) {
        obs[128 + i] = (legal >> i) & 1 ? 1.0f : 0.0f;
    }
}

// Step function for the agent's turn
// Returns: 0 = continue, 1 = terminal
static inline int oth_step_agent(Othello *g, int action, int agent_color) {
    uint64_t my_board = (agent_color == OTH_BLACK) ? g->black : g->white;
    uint64_t opp_board = (agent_color == OTH_BLACK) ? g->white : g->black;
    uint64_t legal = oth_get_moves(my_board, opp_board);

    // Validate action
    if (action == OTH_ACTION_PASS) {
        // Pass is only valid when no legal moves exist
        if (legal != 0) {
            // Invalid pass: there are legal moves available
            g->done = 1;
            g->reward = (agent_color == OTH_BLACK) ? -1.0f : 1.0f;
            g->invalid_moves++;
            return 1;
        }
    } else if (action < 0 || action >= 64) {
        // Out of bounds
        g->done = 1;
        g->reward = (agent_color == OTH_BLACK) ? -1.0f : 1.0f;
        g->invalid_moves++;
        return 1;
    } else {
        // Check if the move is legal
        if (!((1ULL << action) & legal)) {
            g->done = 1;
            g->reward = (agent_color == OTH_BLACK) ? -1.0f : 1.0f;
            g->invalid_moves++;
            return 1;
        }
    }

    // Apply the valid move
    oth_apply_move(g, action);
    return oth_check_terminal(g);
}

#endif // OTHELLO_H
