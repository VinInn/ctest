#define NPY_NO_DEPRECATED_API NPY_1_7_API_VERSION
#include "Python.h"
#include "math.h"
#include "numpy/ndarraytypes.h"
#include "numpy/ufuncobject.h"
#include "numpy/npy_3kcompat.h"

/*
 * single_type_tiwce.c
 * This is the C code for creating your own
 * NumPy ufunc for a YOUR_FUNC function.
 *
 * In this code we only define the ufunc for
 * a single dtype. The computations that must
 * be replaced to create a ufunc for
 * a different function are marked with BEGIN
 * and END.
 *
 * Details explaining the Python-C API can be found under
 * 'Extending and Embedding' and 'Python/C API' at
 * docs.python.org .
 */

#define YOUR_FUNC twice
#define xstr(s) str(s)
#define str(s) #s

static PyMethodDef FuncMethods[] = {
        {NULL, NULL, 0, NULL}
};

/* The loop definition must precede the PyMODINIT_FUNC. */
void YOUR_FUNC(const float * vi, float * vo, long n);

static void YOUR_FUNCf(char **args, npy_intp *dimensions,
                            npy_intp* steps, void* data)
{
/*    npy_intp i;  */
    npy_intp n = dimensions[0];
    char *in = args[0], *out = args[1];
    /* npy_intp in_step = steps[0], out_step = steps[1]; */

    YOUR_FUNC((const float *)in, (float *)out,n); 
}

/*This a pointer to the above function*/
PyUFuncGenericFunction funcs[1] = {&YOUR_FUNCf};

/* These are the input and return dtypes of YOUR_FUNC.*/
static char types[2] = {NPY_FLOAT, NPY_FLOAT};

static void *data[1] = {NULL};

#if PY_VERSION_HEX >= 0x03000000
static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT,
    "npufunc",
    NULL,
    -1,
    FuncMethods,
    NULL,
    NULL,
    NULL,
    NULL
};

PyMODINIT_FUNC PyInit_npufunc(void)
{
    PyObject *m, *YOUR_FUNC, *d;
    m = PyModule_Create(&moduledef);
    if (!m) {
        return NULL;
    }

    import_array();
    import_umath();

    YOUR_FUNC = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, "YOUR_FUNC",
                                    "tiwce_docstring", 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, "YOUR_FUNC", YOUR_FUNC);
    Py_DECREF(YOUR_FUNC);

    return m;
}
#else
PyMODINIT_FUNC initnpufunc(void)
{
    PyObject *m, *YOUR_FUNC, *d;


    m = Py_InitModule("npufunc", FuncMethods);
    if (m == NULL) {
        return;
    }

    import_array();
    import_umath();

    YOUR_FUNC = PyUFunc_FromFuncAndData(funcs, data, types, 1, 1, 1,
                                    PyUFunc_None, xstr(YOUR_FUNC),
                                    xstr(YOUR_FUNC), 0);

    d = PyModule_GetDict(m);

    PyDict_SetItemString(d, xstr(YOUR_FUNC), YOUR_FUNC);
    Py_DECREF(YOUR_FUNC);
}
#endif

