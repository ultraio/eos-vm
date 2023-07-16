#pragma once

#include <eosio/vm/allocator.hpp>
#include <eosio/vm/bitcode_writer.hpp>
#include <eosio/vm/config.hpp>
#include <eosio/vm/debug_visitor.hpp>
#include <eosio/vm/execution_context.hpp>
#include <eosio/vm/interpret_visitor.hpp>
#include <eosio/vm/parser.hpp>
#include <eosio/vm/types.hpp>
#include <eosio/vm/x86_64.hpp>

#include <atomic>
#include <exception>
#include <fstream>
#include <iostream>
#include <optional>
#include <string_view>
#include <system_error>
#include <vector>

namespace eosio { namespace vm {

   struct jit {
      template<typename Host>
      using context = jit_execution_context<Host>;
      template<typename Host>
      using parser = binary_parser<machine_code_writer<jit_execution_context<Host>>>;
      static constexpr bool is_jit = true;
   };

   struct interpreter {
      template<typename Host>
      using context = execution_context<Host>;
      template<typename Host>
      using parser = binary_parser<bitcode_writer>;
      static constexpr bool is_jit = false;
   };

   template <typename Host, typename Impl = interpreter>
   class backend {
    public:
      using host_t = Host;

      template <typename HostFunctions = nullptr_t>
      backend(wasm_code& code, HostFunctions = nullptr) : _ctx(parse_module(code)) {
	 if constexpr (!std::is_same_v<HostFunctions, nullptr_t>)
            HostFunctions::resolve(_mod);
	 _mod.finalize();
      }
      template <typename HostFunctions = nullptr_t>
      backend(wasm_code_ptr& ptr, size_t sz, HostFunctions = nullptr, bool single_parsing = true) : _ctx(parse_module2(ptr, sz, single_parsing)) {
	 if constexpr (!std::is_same_v<HostFunctions, nullptr_t>)
            HostFunctions::resolve(_mod);
	 _mod.finalize();
      }

      module& parse_module(wasm_code& code) {
         _mod.allocator.use_default_memory();
         return typename Impl::template parser<Host>{ _mod.allocator }.parse_module(code, _mod);
      }

      module& parse_module2(wasm_code_ptr& ptr, size_t sz, bool single_parsing) {
         if (single_parsing) {
            _mod.allocator.use_default_memory();
            return typename Impl::template parser<Host>{ _mod.allocator }.parse_module2(ptr, sz, _mod);
         }

         // To prevent large number of memory mappings used, use two-passes parsing.
         // The first pass finds max size of memory required for parsing;
         // this memory is released after parsing.
         // The second pass uses malloc with the required size of memory.
         wasm_code_ptr orig_ptr = ptr;
         size_t largest_size = 0;

         // First pass: finds max size of memory required by parsing.
         // Memory used by parsing will be freed when going out of the scope
         {
            module first_pass_module;
            // For JIT, skips code generation as it is not needed and
            // does not count the code memory size
            detail::code_generate_mode code_gen_mode = Impl::is_jit ? detail::code_generate_mode::skip : detail::code_generate_mode::use_same_allocator;
            first_pass_module.allocator.use_default_memory();
            typename Impl::template parser<Host>{ first_pass_module.allocator }.parse_module2(ptr, sz, first_pass_module, code_gen_mode);
            first_pass_module.finalize();
            largest_size = first_pass_module.allocator.largest_used_size();
         }

         // Second pass: uses largest_size of memory for actual parsing
         _mod.allocator.use_fixed_memory(Impl::is_jit, largest_size);
         // For JIT, uses a seperate allocator for code generation as mod's memory
         // does not include memory for code
         detail::code_generate_mode code_gen_mode = Impl::is_jit ? detail::code_generate_mode::use_seperate_allocator : detail::code_generate_mode::use_same_allocator;
         return typename Impl::template parser<Host>{ _mod.allocator }.parse_module2(orig_ptr, sz, _mod, code_gen_mode);
      }

      template <typename... Args>
      inline bool operator()(Host* host, const std::string_view& mod, const std::string_view& func, Args... args) {
         return call(host, mod, func, args...);
      }

      inline backend& initialize(Host* host=nullptr) {
         _ctx.reset();
         _ctx.execute_start(host, interpret_visitor(_ctx));
         return *this;
      }

      template <typename... Args>
      inline bool call_indirect(Host* host, uint32_t func_index, Args... args) {
         if constexpr (eos_vm_debug) {
            //_ctx.execute_func_table(host, debug_visitor(_ctx), func_index, args...);
            _ctx.execute_func_table(host, interpret_visitor(_ctx), func_index, args...);
         } else {
            _ctx.execute_func_table(host, interpret_visitor(_ctx), func_index, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(Host* host, uint32_t func_index, Args... args) {
         if constexpr (eos_vm_debug) {
            //_ctx.execute(host, debug_visitor(_ctx), func_index, args...);
            _ctx.execute(host, interpret_visitor(_ctx), func_index, args...);
         } else {
            _ctx.execute(host, interpret_visitor(_ctx), func_index, args...);
         }
         return true;
      }

      template <typename... Args>
      inline bool call(Host* host, const std::string_view& mod, const std::string_view& func, Args... args) {
         if constexpr (eos_vm_debug) {
            //_ctx.execute(host, debug_visitor(_ctx), func, args...);
            _ctx.execute(host, interpret_visitor(_ctx), func, args...);
         } else {
            _ctx.execute(host, interpret_visitor(_ctx), func, args...);
         }
         return true;
      }

      template <typename... Args>
      inline auto call_with_return(Host* host, const std::string_view& mod, const std::string_view& func,
                                   Args... args) {
         if constexpr (eos_vm_debug) {
            //return _ctx.execute(host, debug_visitor(_ctx), func, args...);
            return _ctx.execute(host, interpret_visitor(_ctx), func, args...);
         } else {
            return _ctx.execute(host, interpret_visitor(_ctx), func, args...);
         }
      }

      void print_result(const std::optional<operand_stack_elem>& result) {
         if(result) {
            std::cout << "result: ";
            if (result->is_a<i32_const_t>())
               std::cout << "i32:" << result->to_ui32();
            else if (result->is_a<i64_const_t>())
               std::cout << "i64:" << result->to_ui64();
            else if (result->is_a<f32_const_t>())
               std::cout << "f32:" << result->to_f32();
            else if (result->is_a<f64_const_t>())
              std::cout << "f64:" << result->to_f64();
            std::cout << std::endl;
        }
      }

      template<typename Watchdog, typename F>
      void timed_run(Watchdog&& wd, F&& f) {
         std::atomic<bool>       _timed_out = false;
         auto reenable_code = scope_guard{[&](){
            if (_timed_out) {
               _mod.allocator.enable_code(Impl::is_jit);
            }
         }};
         try {
            auto wd_guard = wd.scoped_run([this,&_timed_out]() {
               _timed_out = true;
               _mod.allocator.disable_code();
            });
            static_cast<F&&>(f)();
         } catch(wasm_memory_exception&) {
            if (_timed_out) {
               throw timeout_exception{ "execution timed out" };
            } else {
               throw;
            }
         }
      }

      template <typename Watchdog>
      inline void execute_all(Watchdog&& wd, Host* host = nullptr) {
         timed_run(static_cast<Watchdog&&>(wd), [&]() {
            for (int i = 0; i < _mod.exports.size(); i++) {
               if (_mod.exports[i].kind == external_kind::Function) {
                  std::string s{ (const char*)_mod.exports[i].field_str.raw(), _mod.exports[i].field_str.size() };
	          if constexpr (eos_vm_debug) {
                     print_result(_ctx.execute(host, debug_visitor(_ctx), s));
	          } else {
	             _ctx.execute(host, interpret_visitor(_ctx), s);
	          }
               }
            }
         });
      }

      inline void set_wasm_allocator(wasm_allocator* walloc) {
         _walloc = walloc;
         _ctx.set_wasm_allocator(walloc);
      }

      inline wasm_allocator* get_wasm_allocator() { return _walloc; }
      inline module&         get_module() { return _mod; }
      inline void            exit(const std::error_code& ec) { _ctx.exit(ec); }
      inline auto&           get_context() { return _ctx; }

      static std::vector<uint8_t> read_wasm(const std::string& fname) {
         std::ifstream wasm_file(fname, std::ios::binary);
         if (!wasm_file.is_open())
            throw std::runtime_error("wasm file not found");
         wasm_file.seekg(0, std::ios::end);
         std::vector<uint8_t> wasm;
         int                  len = wasm_file.tellg();
         if (len < 0)
            throw std::runtime_error("wasm file length is -1");
         wasm.resize(len);
         wasm_file.seekg(0, std::ios::beg);
         wasm_file.read((char*)wasm.data(), wasm.size());
         wasm_file.close();
         return wasm;
      }

    private:
      wasm_allocator*         _walloc = nullptr; // non owning pointer
      module                  _mod;
      typename Impl::template context<Host> _ctx;
   };
}} // namespace eosio::vm
