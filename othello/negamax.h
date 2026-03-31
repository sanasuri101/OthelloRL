#ifndef NEGAMAX_H
#define NEGAMAX_H

#include "othello.h"
#include <limits.h>

#define NEG_NODE_LIMIT 1000000
#define NEG_INF 999999

// Classic positional weight table for Othello
// Corners are extremely valuable, X-squares (diagonal to corners) are dangerous
static const int NEG_WEIGHTS[64] = {
    100, -50,  10,   5,   5,  10, -50, 100,
    -50, -50,  -2,  -2,  -2,  -2, -50, -50,
     10,  -2,   1,   1,   1,   1,  -2,  10,
      5,  -2,   1,   1,   1,   1,  -2,   5,
      5,  -2,   1,   1,   1,   1,  -2,   5,
     10,  -2,   1,   1,   1,   1,  -2,  10,
    -50, -50,  -2,  -2,  -2,  -2, -50, -50,
    100, -50,  10,   5,   5,  10, -50, 100
};

typedef struct {
    int node_count;
} NegState;

static inline int neg_evaluate(uint64_t my_board, uint64_t opp_board) {
    int score = 0;

    // Positional weights
    for (int i = 0; i < 64; i++) {
        if ((my_board >> i) & 1) {
            score += NEG_WEIGHTS[i];
        }
        if ((opp_board >> i) & 1) {
            score -= NEG_WEIGHTS[i];
        }
    }

    // Mobility component: number of moves available
    int my_mobility = popcount64(oth_get_moves(my_board, opp_board));
    int opp_mobility = popcount64(oth_get_moves(opp_board, my_board));
    score += (my_mobility - opp_mobility) * 5;

    return score;
}

static inline int neg_search(
    uint64_t my_board, uint64_t opp_board,
    int depth, int alpha, int beta,
    int pass_count, NegState *state
) {
    state->node_count++;

    // Check termination conditions
    if (state->node_count >= NEG_NODE_LIMIT || depth <= 0 || pass_count >= 2) {
        return neg_evaluate(my_board, opp_board);
    }

    uint64_t moves = oth_get_moves(my_board, opp_board);

    if (moves == 0) {
        // Must pass -- recurse with swapped boards and incremented pass count
        return -neg_search(opp_board, my_board, depth - 1, -beta, -alpha,
                           pass_count + 1, state);
    }

    int best = -NEG_INF;

    while (moves) {
        int sq = __builtin_ctzll(moves);
        moves &= moves - 1;  // clear lowest set bit

        // Compute flips for this move
        uint64_t flips = oth_resolve_flips(my_board, opp_board, sq);
        uint64_t new_my = my_board | (1ULL << sq) | flips;
        uint64_t new_opp = opp_board & ~flips;

        int val = -neg_search(new_opp, new_my, depth - 1, -beta, -alpha, 0, state);

        if (val > best) {
            best = val;
        }
        if (val > alpha) {
            alpha = val;
        }
        if (alpha >= beta) {
            break;  // beta cutoff
        }

        if (state->node_count >= NEG_NODE_LIMIT) {
            break;
        }
    }

    return best;
}

// Returns best move (0-63) or OTH_ACTION_PASS (64) if no moves available
static inline int neg_best_move(Othello *g, int color, int depth) {
    uint64_t my_board = (color == OTH_BLACK) ? g->black : g->white;
    uint64_t opp_board = (color == OTH_BLACK) ? g->white : g->black;
    uint64_t moves = oth_get_moves(my_board, opp_board);

    if (moves == 0) {
        return OTH_ACTION_PASS;
    }

    NegState state = { .node_count = 0 };
    int best_score = -NEG_INF;
    int best_move = OTH_ACTION_PASS;

    while (moves) {
        int sq = __builtin_ctzll(moves);
        moves &= moves - 1;

        uint64_t flips = oth_resolve_flips(my_board, opp_board, sq);
        uint64_t new_my = my_board | (1ULL << sq) | flips;
        uint64_t new_opp = opp_board & ~flips;

        int val = -neg_search(new_opp, new_my, depth - 1, -NEG_INF, NEG_INF, 0, &state);

        if (val > best_score) {
            best_score = val;
            best_move = sq;
        }

        if (state.node_count >= NEG_NODE_LIMIT) {
            break;
        }
    }

    return best_move;
}

#endif // NEGAMAX_H
