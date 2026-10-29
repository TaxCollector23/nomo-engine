// nomo._core: C++17 kernels with semantics identical to nomo/runtime/qgraph.py.
//
// lif_layer(W[No,Ni] int8, b[No] int32, s_in[T,Ni] int8, theta, leak_shift, v_bits) -> s_out[T,No] int8
//   per step t:  v -= asr(v, k) (k > 0);  v = clamp(v + W s_t + b, v_bits);  z = v >= theta;  v -= theta z
//
// Event-driven inner loop: only columns with a non-zero input spike are touched, so work scales
// with the synaptic-op count the cost model charges (rate x N_in x fan-out) rather than N_in x N_out.
#include <pybind11/numpy.h>
#include <pybind11/pybind11.h>

#include <cstdint>
#include <stdexcept>
#include <vector>

namespace py = pybind11;

static inline int64_t asr64(int64_t v, int n) { return v < 0 ? ~((~v) >> n) : (v >> n); }

py::array_t<int8_t> lif_layer(py::array_t<int8_t, py::array::c_style | py::array::forcecast> W,
                              py::array_t<int32_t, py::array::c_style | py::array::forcecast> b,
                              py::array_t<int8_t, py::array::c_style | py::array::forcecast> s_in,
                              int64_t theta, int leak_shift, int v_bits) {
  if (W.ndim() != 2 || b.ndim() != 1 || s_in.ndim() != 2) throw std::invalid_argument("bad ranks");
  const ssize_t No = W.shape(0), Ni = W.shape(1), T = s_in.shape(0);
  if (b.shape(0) != No || s_in.shape(1) != Ni) throw std::invalid_argument("shape mismatch");
  if (v_bits < 2 || v_bits > 32) throw std::invalid_argument("v_bits out of range");
  const int64_t lo = -(int64_t(1) << (v_bits - 1)), hi = (int64_t(1) << (v_bits - 1)) - 1;

  auto w = W.unchecked<2>();
  auto bb = b.unchecked<1>();
  auto s = s_in.unchecked<2>();
  py::array_t<int8_t> out({T, No});
  auto o = out.mutable_unchecked<2>();

  std::vector<int64_t> v(No, 0), inc(No, 0);
  std::vector<ssize_t> active;
  active.reserve(Ni);
  {
    py::gil_scoped_release release;
    for (ssize_t t = 0; t < T; ++t) {
      active.clear();
      for (ssize_t i = 0; i < Ni; ++i)
        if (s(t, i) != 0) active.push_back(i);
      for (ssize_t j = 0; j < No; ++j) {
        int64_t acc = bb(j);
        for (ssize_t i : active) acc += int64_t(w(j, i)) * int64_t(s(t, i));
        inc[j] = acc;
      }
      for (ssize_t j = 0; j < No; ++j) {
        if (inc[j] > INT32_MAX || inc[j] < INT32_MIN) throw std::overflow_error("LIF synaptic accumulator overflow");
        int64_t vj = v[j];
        if (leak_shift > 0) vj -= asr64(vj, leak_shift);
        vj += inc[j];
        vj = vj < lo ? lo : (vj > hi ? hi : vj);
        const int8_t z = vj >= theta ? 1 : 0;
        vj -= theta * z;
        v[j] = vj;
        o(t, j) = z;
      }
    }
  }
  return out;
}

PYBIND11_MODULE(_core, m) {
  m.doc() = "Nomo C++ kernels (bit-exact with the Python golden model)";
  m.def("lif_layer", &lif_layer, py::arg("W"), py::arg("b"), py::arg("s_in"), py::arg("theta"),
        py::arg("leak_shift"), py::arg("v_bits"));
}
