#ifndef RENDER_H
#define RENDER_H

#include "othello.h"

#ifndef NO_RENDER
#include "raylib.h"
#endif

#define WINDOW_WIDTH 600
#define WINDOW_HEIGHT 700
#define HEADER_HEIGHT 50
#define FOOTER_HEIGHT 50
#define BOARD_SIZE_PX (WINDOW_HEIGHT - HEADER_HEIGHT - FOOTER_HEIGHT)
#define CELL_SIZE (BOARD_SIZE_PX / 8)
#define BOARD_OFFSET_X ((WINDOW_WIDTH - BOARD_SIZE_PX) / 2)
#define BOARD_OFFSET_Y HEADER_HEIGHT
#define PIECE_RADIUS (CELL_SIZE / 2 - 6)

#ifdef NO_RENDER

/* Headless stubs — compiled when NO_RENDER is defined (e.g. in CI). */
static void render_board(Othello *g) { (void)g; }
static int  render_should_close(void) { return 0; }
static int  render_get_click(void) { return -1; }

#else /* full raylib implementation */

static int render_initialized = 0;

static void render_init(void) {
    if (render_initialized) return;
    InitWindow(WINDOW_WIDTH, WINDOW_HEIGHT, "Othello RL");
    SetTargetFPS(60);
    render_initialized = 1;
}

static void render_close(void) {
    if (!render_initialized) return;
    CloseWindow();
    render_initialized = 0;
}

static void render_board(Othello *g) {
    if (!render_initialized) render_init();

    BeginDrawing();
    ClearBackground((Color){30, 30, 46, 255});

    /* Header */
    DrawText(TextFormat("Move %d", g->move_count), 20, 15, 20, LIGHTGRAY);
    int bc = popcount64(g->black);
    int wc = popcount64(g->white);
    DrawText(TextFormat("Black %d  -  White %d", bc, wc),
             WINDOW_WIDTH - 220, 15, 20, LIGHTGRAY);

    /* Board background */
    DrawRectangle(BOARD_OFFSET_X, BOARD_OFFSET_Y,
                  BOARD_SIZE_PX, BOARD_SIZE_PX, (Color){21, 128, 61, 255});

    /* Compute legal moves for the current player */
    uint64_t mine = (g->current_player == OTH_BLACK) ? g->black : g->white;
    uint64_t opp  = (g->current_player == OTH_BLACK) ? g->white : g->black;
    uint64_t legal = oth_get_moves(mine, opp);

    /* Grid lines and cells */
    for (int row = 0; row < 8; row++) {
        for (int col = 0; col < 8; col++) {
            int x = BOARD_OFFSET_X + col * CELL_SIZE;
            int y = BOARD_OFFSET_Y + row * CELL_SIZE;
            DrawRectangleLines(x, y, CELL_SIZE, CELL_SIZE,
                               (Color){0, 80, 30, 255});

            int sq = row * 8 + col;
            int cx = x + CELL_SIZE / 2;
            int cy = y + CELL_SIZE / 2;

            /* Draw pieces with 3D shading */
            if (g->black & (1ULL << sq)) {
                DrawCircle(cx, cy, PIECE_RADIUS,
                           (Color){34, 34, 34, 255});
                DrawCircle(cx - 2, cy - 2, PIECE_RADIUS - 3,
                           (Color){50, 50, 50, 255});
            } else if (g->white & (1ULL << sq)) {
                DrawCircle(cx, cy, PIECE_RADIUS,
                           (Color){220, 220, 220, 255});
                DrawCircle(cx - 2, cy - 2, PIECE_RADIUS - 3,
                           (Color){240, 240, 240, 255});
            }

            /* Valid move indicators */
            if (legal & (1ULL << sq)) {
                DrawCircleLines(cx, cy, 10, (Color){255, 255, 255, 80});
            }
        }
    }

    /* Footer */
    DrawText("ESC to quit", WINDOW_WIDTH - 130, WINDOW_HEIGHT - 35, 16, GRAY);

    EndDrawing();
}

static int render_should_close(void) {
    return render_initialized ? WindowShouldClose() : 0;
}

static int render_get_click(void) {
    if (!render_initialized) return -1;
    if (!IsMouseButtonPressed(MOUSE_LEFT_BUTTON)) return -1;
    int mx = GetMouseX() - BOARD_OFFSET_X;
    int my = GetMouseY() - BOARD_OFFSET_Y;
    if (mx < 0 || mx >= BOARD_SIZE_PX || my < 0 || my >= BOARD_SIZE_PX) return -1;
    int col = mx / CELL_SIZE;
    int row = my / CELL_SIZE;
    return row * 8 + col;
}

#endif /* NO_RENDER */

#endif /* RENDER_H */
