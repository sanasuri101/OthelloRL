#define PY_SSIZE_T_CLEAN
#include <Python.h>
#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include <numpy/arrayobject.h>
#include <stdlib.h>
#include <time.h>

#include "othello.h"
#include "negamax.h"
#include "render.h"

typedef struct {
    PyObject_HEAD
    Othello *envs;
    int num_envs;
    float **obs_ptrs;
    int **action_ptrs;
    float **reward_ptrs;
    int **done_ptrs;
    // Stats accumulators
    int total_wins;
    int total_games;
    int total_invalid_moves;
    int total_moves;
    int total_corner_captures;
} VecEnv;

static void VecEnv_dealloc(VecEnv *self) {
    if (self->envs) free(self->envs);
    if (self->obs_ptrs) free(self->obs_ptrs);
    if (self->action_ptrs) free(self->action_ptrs);
    if (self->reward_ptrs) free(self->reward_ptrs);
    if (self->done_ptrs) free(self->done_ptrs);
    Py_TYPE(self)->tp_free((PyObject *)self);
}

static PyObject *VecEnv_new(PyTypeObject *type, PyObject *args, PyObject *kwds) {
    (void)args; (void)kwds;
    VecEnv *self = (VecEnv *)type->tp_alloc(type, 0);
    if (self) {
        self->envs = NULL;
        self->num_envs = 0;
        self->obs_ptrs = NULL;
        self->action_ptrs = NULL;
        self->reward_ptrs = NULL;
        self->done_ptrs = NULL;
        self->total_wins = 0;
        self->total_games = 0;
        self->total_invalid_moves = 0;
        self->total_moves = 0;
        self->total_corner_captures = 0;
    }
    return (PyObject *)self;
}

static PyObject *vec_init(VecEnv *self, PyObject *args) {
    int num_envs;
    PyObject *obs_arr, *actions_arr, *rewards_arr, *dones_arr;

    if (!PyArg_ParseTuple(args, "iOOOO", &num_envs, &obs_arr, &actions_arr,
                          &rewards_arr, &dones_arr)) {
        return NULL;
    }

    self->num_envs = num_envs;
    self->envs = (Othello *)calloc(num_envs, sizeof(Othello));
    self->obs_ptrs = (float **)calloc(num_envs, sizeof(float *));
    self->action_ptrs = (int **)calloc(num_envs, sizeof(int *));
    self->reward_ptrs = (float **)calloc(num_envs, sizeof(float *));
    self->done_ptrs = (int **)calloc(num_envs, sizeof(int *));

    if (!self->envs || !self->obs_ptrs || !self->action_ptrs ||
        !self->reward_ptrs || !self->done_ptrs) {
        PyErr_SetString(PyExc_MemoryError, "Failed to allocate environment memory");
        return NULL;
    }

    // Slice numpy buffers per-env
    float *obs_base = (float *)PyArray_DATA((PyArrayObject *)obs_arr);
    int *act_base = (int *)PyArray_DATA((PyArrayObject *)actions_arr);
    float *rew_base = (float *)PyArray_DATA((PyArrayObject *)rewards_arr);
    // dones are stored as uint8 in pufferlib but we treat as int pointers
    // Actually pufferlib uses bool/uint8 for dones, let's handle properly
    int *done_base = (int *)PyArray_DATA((PyArrayObject *)dones_arr);

    for (int i = 0; i < num_envs; i++) {
        self->obs_ptrs[i] = obs_base + i * OTH_OBS_DIM;
        self->action_ptrs[i] = act_base + i;
        self->reward_ptrs[i] = rew_base + i;
        self->done_ptrs[i] = done_base + i;
    }

    srand((unsigned int)time(NULL));

    Py_RETURN_NONE;
}

static PyObject *vec_reset(VecEnv *self, PyObject *args) {
    (void)args;
    for (int i = 0; i < self->num_envs; i++) {
        oth_reset(&self->envs[i]);
        self->envs[i].episode_id = i;  // Initial episode ID
        int agent_color = self->envs[i].episode_id % 2;
        oth_write_obs(&self->envs[i], self->obs_ptrs[i], agent_color);
        *self->reward_ptrs[i] = 0.0f;
        *self->done_ptrs[i] = 0;

        // If agent is white, opponent (black) moves first
        if (agent_color == OTH_WHITE) {
            uint64_t opp_moves = oth_get_moves(self->envs[i].black, self->envs[i].white);
            if (opp_moves) {
                // Pick random opponent move
                int count = popcount64(opp_moves);
                int choice = rand() % count;
                int sq = 0;
                for (int b = 0; b < 64; b++) {
                    if ((opp_moves >> b) & 1) {
                        if (choice == 0) { sq = b; break; }
                        choice--;
                    }
                }
                oth_apply_move(&self->envs[i], sq);
            }
            oth_write_obs(&self->envs[i], self->obs_ptrs[i], agent_color);
        }
    }
    Py_RETURN_NONE;
}

static inline void play_random_opponent(Othello *g, int opp_color) {
    uint64_t opp_board = (opp_color == OTH_BLACK) ? g->black : g->white;
    uint64_t my_board = (opp_color == OTH_BLACK) ? g->white : g->black;
    uint64_t opp_moves = oth_get_moves(opp_board, my_board);

    if (opp_moves == 0) {
        // Opponent must pass
        oth_apply_move(g, OTH_ACTION_PASS);
    } else {
        int count = popcount64(opp_moves);
        int choice = rand() % count;
        int sq = 0;
        for (int b = 0; b < 64; b++) {
            if ((opp_moves >> b) & 1) {
                if (choice == 0) { sq = b; break; }
                choice--;
            }
        }
        oth_apply_move(g, sq);
    }
}

static PyObject *vec_step(VecEnv *self, PyObject *args) {
    (void)args;
    for (int i = 0; i < self->num_envs; i++) {
        Othello *g = &self->envs[i];
        int agent_color = g->episode_id % 2;

        // Auto-reset if done
        if (g->done) {
            // Accumulate stats before reset
            self->total_games++;
            self->total_moves += g->move_count;
            self->total_invalid_moves += g->invalid_moves;
            self->total_corner_captures += g->corner_captures;
            float terminal_reward = g->reward;
            if (agent_color == OTH_WHITE) {
                terminal_reward = -terminal_reward;  // flip for white agent
            }
            if (terminal_reward > 0) {
                self->total_wins++;
            }

            // Reset and increment episode
            oth_reset(g);
            g->episode_id++;
            agent_color = g->episode_id % 2;

            // If agent is white, opponent (black) goes first
            if (agent_color == OTH_WHITE) {
                play_random_opponent(g, OTH_BLACK);
                if (oth_check_terminal(g)) {
                    oth_write_obs(g, self->obs_ptrs[i], agent_color);
                    *self->reward_ptrs[i] = (agent_color == OTH_WHITE) ? -g->reward : g->reward;
                    *self->done_ptrs[i] = 1;
                    continue;
                }
            }

            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = 0.0f;
            *self->done_ptrs[i] = 0;
            continue;
        }

        int action = *self->action_ptrs[i];

        // Agent's turn
        int terminal = oth_step_agent(g, action, agent_color);
        if (terminal) {
            float r = g->reward;
            if (agent_color == OTH_WHITE) {
                r = -r;  // reward is stored from black's perspective
            }
            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = r;
            *self->done_ptrs[i] = 1;
            continue;
        }

        // Opponent's turn
        int opp_color = 1 - agent_color;
        play_random_opponent(g, opp_color);

        // Check terminal after opponent
        if (oth_check_terminal(g)) {
            float r = g->reward;
            if (agent_color == OTH_WHITE) {
                r = -r;
            }
            oth_write_obs(g, self->obs_ptrs[i], agent_color);
            *self->reward_ptrs[i] = r;
            *self->done_ptrs[i] = 1;
            continue;
        }

        // Check if agent has moves; if not, auto-pass for agent
        uint64_t agent_board = (agent_color == OTH_BLACK) ? g->black : g->white;
        uint64_t enemy_board = (agent_color == OTH_BLACK) ? g->white : g->black;
        uint64_t agent_moves = oth_get_moves(agent_board, enemy_board);

        if (agent_moves == 0) {
            // Agent must pass
            oth_apply_move(g, OTH_ACTION_PASS);
            if (oth_check_terminal(g)) {
                float r = g->reward;
                if (agent_color == OTH_WHITE) {
                    r = -r;
                }
                oth_write_obs(g, self->obs_ptrs[i], agent_color);
                *self->reward_ptrs[i] = r;
                *self->done_ptrs[i] = 1;
                continue;
            }

            // Opponent plays again
            play_random_opponent(g, opp_color);
            if (oth_check_terminal(g)) {
                float r = g->reward;
                if (agent_color == OTH_WHITE) {
                    r = -r;
                }
                oth_write_obs(g, self->obs_ptrs[i], agent_color);
                *self->reward_ptrs[i] = r;
                *self->done_ptrs[i] = 1;
                continue;
            }
        }

        // Write observations
        oth_write_obs(g, self->obs_ptrs[i], agent_color);
        *self->reward_ptrs[i] = 0.0f;
        *self->done_ptrs[i] = 0;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_close(VecEnv *self, PyObject *args) {
    (void)args;
    if (self->envs) {
        free(self->envs);
        self->envs = NULL;
    }
    Py_RETURN_NONE;
}

static PyObject *vec_log(VecEnv *self, PyObject *args) {
    (void)args;
    PyObject *dict = PyDict_New();
    if (!dict) return NULL;

    double win_rate = (self->total_games > 0) ?
        (double)self->total_wins / self->total_games : 0.0;
    double avg_length = (self->total_games > 0) ?
        (double)self->total_moves / self->total_games : 0.0;
    double invalid_rate = (self->total_moves > 0) ?
        (double)self->total_invalid_moves / self->total_moves : 0.0;

    PyDict_SetItemString(dict, "win_rate", PyFloat_FromDouble(win_rate));
    PyDict_SetItemString(dict, "avg_game_length", PyFloat_FromDouble(avg_length));
    PyDict_SetItemString(dict, "invalid_move_rate", PyFloat_FromDouble(invalid_rate));
    PyDict_SetItemString(dict, "games_played", PyLong_FromLong(self->total_games));
    PyDict_SetItemString(dict, "corner_captures", PyLong_FromLong(self->total_corner_captures));

    // Reset accumulators
    self->total_wins = 0;
    self->total_games = 0;
    self->total_invalid_moves = 0;
    self->total_moves = 0;
    self->total_corner_captures = 0;

    return dict;
}

static PyObject *vec_negamax_move(VecEnv *self, PyObject *args) {
    int env_idx, depth;
    if (!PyArg_ParseTuple(args, "ii", &env_idx, &depth)) {
        return NULL;
    }
    if (env_idx < 0 || env_idx >= self->num_envs) {
        PyErr_SetString(PyExc_IndexError, "Environment index out of range");
        return NULL;
    }
    Othello *g = &self->envs[env_idx];
    int color = g->current_player;
    int move = neg_best_move(g, color, depth);
    return PyLong_FromLong(move);
}

static PyObject *vec_apply_opponent_move(VecEnv *self, PyObject *args) {
    int env_idx, move;
    if (!PyArg_ParseTuple(args, "ii", &env_idx, &move)) {
        return NULL;
    }
    if (env_idx < 0 || env_idx >= self->num_envs) {
        PyErr_SetString(PyExc_IndexError, "Environment index out of range");
        return NULL;
    }
    oth_apply_move(&self->envs[env_idx], move);
    Py_RETURN_NONE;
}

static PyObject *vec_test_init(VecEnv *self, PyObject *args) {
    (void)self; (void)args;
    Othello g;
    oth_reset(&g);
    int black_count = popcount64(g.black);
    int white_count = popcount64(g.white);
    uint64_t legal = oth_get_moves(g.black, g.white);
    int legal_count = popcount64(legal);
    return Py_BuildValue("(iii)", black_count, white_count, legal_count);
}

static PyObject *vec_render_should_close(VecEnv *self, PyObject *args) {
    (void)self; (void)args;
    return PyBool_FromLong(render_should_close());
}

static PyObject *vec_render_get_click(VecEnv *self, PyObject *args) {
    (void)self; (void)args;
    return PyLong_FromLong(render_get_click());
}

static PyObject *vec_render(VecEnv *self, PyObject *args) {
    int env_idx;
    if (!PyArg_ParseTuple(args, "i", &env_idx)) {
        return NULL;
    }
    if (env_idx < 0 || env_idx >= self->num_envs) {
        PyErr_SetString(PyExc_IndexError, "Environment index out of range");
        return NULL;
    }
    render_board(&self->envs[env_idx]);
    Py_RETURN_NONE;
}

static PyMethodDef VecEnv_methods[] = {
    {"init", (PyCFunction)vec_init, METH_VARARGS, "Initialize environments"},
    {"reset", (PyCFunction)vec_reset, METH_NOARGS, "Reset all environments"},
    {"step", (PyCFunction)vec_step, METH_NOARGS, "Step all environments"},
    {"close", (PyCFunction)vec_close, METH_NOARGS, "Free environment memory"},
    {"log", (PyCFunction)vec_log, METH_NOARGS, "Get logging statistics"},
    {"negamax_move", (PyCFunction)vec_negamax_move, METH_VARARGS, "Get negamax move"},
    {"apply_opponent_move", (PyCFunction)vec_apply_opponent_move, METH_VARARGS, "Apply opponent move"},
    {"test_init", (PyCFunction)vec_test_init, METH_NOARGS, "Test board initialization"},
    {"render_should_close", (PyCFunction)vec_render_should_close, METH_NOARGS, "Check render close"},
    {"render_get_click", (PyCFunction)vec_render_get_click, METH_NOARGS, "Get render click"},
    {"render", (PyCFunction)vec_render, METH_VARARGS, "Render environment"},
    {NULL, NULL, 0, NULL}
};

static PyTypeObject VecEnvType = {
    PyVarObject_HEAD_INIT(NULL, 0)
    .tp_name = "binding.VecEnv",
    .tp_doc = "Vectorized Othello environment",
    .tp_basicsize = sizeof(VecEnv),
    .tp_itemsize = 0,
    .tp_flags = Py_TPFLAGS_DEFAULT,
    .tp_new = VecEnv_new,
    .tp_dealloc = (destructor)VecEnv_dealloc,
    .tp_methods = VecEnv_methods,
};

static PyMethodDef module_methods[] = {
    {NULL, NULL, 0, NULL}
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "binding",
    "Othello C extension for PufferLib",
    -1,
    module_methods,
};

PyMODINIT_FUNC PyInit_binding(void) {
    import_array();

    PyObject *m = PyModule_Create(&moduledef);
    if (!m) return NULL;

    if (PyType_Ready(&VecEnvType) < 0) return NULL;
    Py_INCREF(&VecEnvType);
    if (PyModule_AddObject(m, "VecEnv", (PyObject *)&VecEnvType) < 0) {
        Py_DECREF(&VecEnvType);
        Py_DECREF(m);
        return NULL;
    }

    // Export constants
    PyModule_AddIntConstant(m, "OBS_DIM", OTH_OBS_DIM);
    PyModule_AddIntConstant(m, "NUM_ACTIONS", OTH_NUM_ACTIONS);

    return m;
}
