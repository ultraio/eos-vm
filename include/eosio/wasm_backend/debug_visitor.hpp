#pragma once

#include <eosio/wasm_backend/interpret_visitor.hpp>

namespace eosio { namespace wasm_backend {

template <typename Backend>
struct debug_visitor : public interpret_visitor<Backend> {
   using interpret_visitor<Backend>::interpret_visitor;

   CONTROL_FLOW_OPS(DBG_VISIT)
   BR_TABLE_OP(DBG_VISIT)
   RETURN_OP(DBG_VISIT)
   CALL_OPS(DBG_VISIT)
   PARAMETRIC_OPS(DBG_VISIT)
   VARIABLE_ACCESS_OPS(DBG_VISIT)
   MEMORY_OPS(DBG_VISIT)
   I32_CONSTANT_OPS(DBG_VISIT)
   I64_CONSTANT_OPS(DBG_VISIT)
   F32_CONSTANT_OPS(DBG_VISIT)
   F64_CONSTANT_OPS(DBG_VISIT)
   COMPARISON_OPS(DBG_VISIT)
   NUMERIC_OPS(DBG_VISIT)
   CONVERSION_OPS(DBG_VISIT)
   SYNTHETIC_OPS(DBG_VISIT)
   ERROR_OPS(DBG_VISIT)
};

}} // ns eosio::wasm_backend
