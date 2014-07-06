// <future> -*- C++ -*-

// Copyright (C) 2009 Free Software Foundation, Inc.
//
// This file is part of the GNU ISO C++ Library.  This library is free
// software; you can redistribute it and/or modify it under the
// terms of the GNU General Public License as published by the
// Free Software Foundation; either version 3, or (at your option)
// any later version.

// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
// GNU General Public License for more details.

// Under Section 7 of GPL version 3, you are granted additional
// permissions described in the GCC Runtime Library Exception, version
// 3.1, as published by the Free Software Foundation.

// You should have received a copy of the GNU General Public License and
// a copy of the GCC Runtime Library Exception along with this program;
// see the files COPYING3 and COPYING.RUNTIME respectively.  If not, see
// <http://www.gnu.org/licenses/>.

/** @file future
 *  This is a Standard C++ Library header.
 */

#ifndef _GLIBCXX_FUTURE
#define _GLIBCXX_FUTURE 1

#pragma GCC system_header

#ifndef __GXX_EXPERIMENTAL_CXX0X__
# include <c++0x_warning.h>
#else

#include <functional>
#include <memory>
#include <mutex>
#include <condition_variable>
#include <system_error>
#include <exception>
#include <cstdatomic>

namespace std
{
  /**
   * @defgroup futures Futures
   * @ingroup concurrency
   *
   * Classes for futures support.
   * @{
   */

  /// Error code for futures
  enum class future_errc
  { broken_promise, future_already_retrieved, promise_already_satisfied };

  // TODO: requires concepts
  // concept_map ErrorCodeEnum<future_errc> { }
  template<>
    struct is_error_code_enum<future_errc> : public true_type { };

  /// Points to a statically-allocated object derived from error_category.
  extern const error_category* const future_category;

  // TODO: requires constexpr
  inline error_code make_error_code(future_errc __errc)
  { return error_code(static_cast<int>(__errc), *future_category); }

  // TODO: requires constexpr
  inline error_condition make_error_condition(future_errc __errc)
  { return error_condition(static_cast<int>(__errc), *future_category); }

  /**
   *  @brief Exception type thrown by futures.
   *  @ingroup exceptions
   */
  class future_error : public logic_error
  {
    error_code _M_code;

  public:
    explicit future_error(future_errc __ec)
    : logic_error("std::future_error"), _M_code(make_error_code(__ec))
    { }

    virtual ~future_error() throw();

    virtual const char* 
    what() const throw();

    const error_code& 
    code() const throw() { return _M_code; }
  };


  inline void __throw_future_error(future_errc __ec) {
    throw future_error(__ec);
  }

  // Forward declarations.
  template<typename _Result>
    class unique_future;

  template<typename _Result>
    class shared_future;

  template<typename> 
    class packaged_task;

  template<typename _Result>
    class promise;

#if defined(_GLIBCXX_HAS_GTHREADS) && defined(_GLIBCXX_USE_C99_STDINT_TR1) \
  && defined(_GLIBCXX_ATOMIC_BUILTINS_4)

  // Holds the result of a future
  struct _Future_result_base
  {
    _Future_result_base() = default;
    _Future_result_base(const _Future_result_base&) = delete;
    _Future_result_base& operator=(const _Future_result_base&) = delete;

    exception_ptr _M_error;

    // _M_destroy() allows derived classes to control deallocation,
    // which will be needed when allocator support is added to promise.
    // See http://gcc.gnu.org/ml/libstdc++/2009-06/msg00032.html
    virtual void _M_destroy() = 0;
    struct _Deleter
    {
      void operator()(_Future_result_base* __fr) const { __fr->_M_destroy(); }
    };

  protected:
    ~_Future_result_base() = default;
  };

  // TODO: use template alias when available
  /*
   template<typename _Res>
     using _Future_ptr = unique_ptr<_Res, _Future_result_base::_Deleter>;
   */
  template<typename _Res>
    struct _Future_ptr
    {
      typedef unique_ptr<_Res, _Future_result_base::_Deleter> type;
    };

  // State shared between a promise and one or more associated futures.
  class _Future_state
  {
    typedef _Future_ptr<_Future_result_base>::type _Future_ptr_type;

  public:
    _Future_state() : _M_result(), _M_retrieved(false) { }

    _Future_state(const _Future_state&) = delete;
    _Future_state& operator=(const _Future_state&) = delete;

    bool
    is_ready()
    { return _M_get() != 0; }

    bool
    has_exception()
    {
      _Future_result_base* const __res = _M_get();
      return __res && !(__res->_M_error == exception_ptr());
    }

    bool
    has_value()
    {
      _Future_result_base* const __res = _M_get();
      return __res   && (__res->_M_error == exception_ptr());
    }

    _Future_result_base&
    wait()
    {
      unique_lock<mutex> __lock(_M_mutex);
      if (!_M_ready())
        _M_cond.wait(__lock, std::bind(&_Future_state::_M_ready, this));
      return *_M_result;
    }

    template<typename _Rep, typename _Period>
      bool
      wait_for(const chrono::duration<_Rep, _Period>& __rel)
      {
        unique_lock<mutex> __lock(_M_mutex);
        return _M_ready() || _M_cond.wait_for(__lock, __rel,
            std::bind(&_Future_state::_M_ready, this));
      }

    template<typename _Clock, typename _Duration>
      bool
      wait_until(const chrono::time_point<_Clock, _Duration>& __abs)
      {
        unique_lock<mutex> __lock(_M_mutex);
        return _M_ready() || _M_cond.wait_until(__lock, __abs,
            std::bind(&_Future_state::_M_ready, this));
      }

    void
    _M_set_result(_Future_ptr_type __res)
    {
      {
        lock_guard<mutex> __lock(_M_mutex);
        if (_M_ready())
	  __throw_future_error(future_errc::promise_already_satisfied);
        _M_result.swap(__res);
      }
      _M_cond.notify_all();
    }

    void
    _M_break_promise(_Future_ptr_type __res)
    {
      if (static_cast<bool>(__res))
      {
        __res->_M_error
          = std::copy_exception(future_error(future_errc::broken_promise));
        {
          lock_guard<mutex> __lock(_M_mutex);
          _M_result.swap(__res);
        }
        _M_cond.notify_all();
      }
    }

    // called when this object is passed to a unique_future
    void
    _M_set_retrieved_flag()
    {
      if (_M_retrieved.test_and_set())
        __throw_future_error(future_errc::future_already_retrieved);
    }

  private:
    _Future_result_base*
    _M_get()
    {
      lock_guard<mutex> __lock(_M_mutex);
      return _M_result.get();
    }

    bool _M_ready() const { return static_cast<bool>(_M_result); }

    _Future_ptr_type    _M_result;
    mutex               _M_mutex;
    condition_variable  _M_cond;
    atomic_flag         _M_retrieved;
  };

  // workaround for CWG issue 664 and c++/34022
  template<typename _Result, bool = is_scalar<_Result>::value>
    struct _Move_future_result
    {
      typedef _Result&& __rval_type;
      static _Result&& _S_move(_Result& __res) { return std::move(__res); }
    };

  // specialization for scalar types returns rvalue not rvalue-reference
  template<typename _Result>
    struct _Move_future_result<_Result, true>
    {
      typedef _Result __rval_type;
      static _Result _S_move(_Result __res) { return __res; }
    };

  template<typename _Result>
    struct _Future_result : _Future_result_base
    {
      _Future_result() : _M_initialized() { }

      ~_Future_result()
      {
        if (_M_initialized)
          _M_value().~_Result();
      }

      // return lvalue, future will add const or rvalue-reference
      _Result& _M_value()
      { return *static_cast<_Result*>(_M_addr()); }

      void
      _M_set(const _Result& __res)
      {
        ::new (_M_addr()) _Result(__res);
        _M_initialized = true;
      }

      void
      _M_set(_Result&& __res)
      {
        typedef _Move_future_result<_Result> _Mover;
        ::new (_M_addr()) _Result(_Mover::_S_move(__res));
        _M_initialized = true;
      }

    private:
      void _M_destroy() { delete this; }

      void* _M_addr() { return static_cast<void*>(&_M_storage); }

      typename aligned_storage<sizeof(_Result),
               alignment_of<_Result>::value>::type _M_storage;
      bool _M_initialized;
    };

  template<typename _Result>
    struct _Future_result<_Result&> : _Future_result_base
    {
      _Future_result() : _M_value_ptr() { }

      _Result* _M_value_ptr;

      void _M_destroy() { delete this; }
    };

  template<>
    struct _Future_result<void> : _Future_result_base
    {
      void _M_destroy() { delete this; }
    };

  // common implementation for unique_future and shared_future
  template<typename _Result>
    class _Future_impl
    {
    public:
      // disable copying
      _Future_impl(const _Future_impl&) = delete;
      _Future_impl& operator=(const _Future_impl&) = delete;

      // functions to check state and wait for ready
      bool is_ready() const { return this->_M_state->is_ready(); }

      bool has_exception() const { return this->_M_state->has_exception(); }

      bool has_value() const { return this->_M_state->has_value(); }

      void wait() const { this->_M_state->wait(); }

      template<typename _Rep, typename _Period>
        bool
        wait_for(const chrono::duration<_Rep, _Period>& __rel) const
        { return this->_M_state->wait_for(__rel); }

      template<typename _Clock, typename _Duration>
        bool
        wait_until(const chrono::time_point<_Clock, _Duration>& __abs) const
        { return this->_M_state->wait_until(__abs); }

    protected:
      // wait for the state to be ready and rethrow any stored exception
      _Future_result<_Result>&
      _M_get_result()
      {
        _Future_result_base& __res = this->_M_state->wait();
        if (!(__res._M_error ==  exception_ptr()))
           rethrow_exception(__res._M_error);
        return static_cast<_Future_result<_Result>&>(__res);
      }

      typedef shared_ptr<_Future_state> _State_ptr;

      // construction of a unique_future by promise::get_future()
      explicit
      _Future_impl(const _State_ptr& __state)
      : _M_state(__state)
      {
        if (static_cast<bool>(this->_M_state))
          this->_M_state->_M_set_retrieved_flag();
        else
          __throw_future_error(future_errc::future_already_retrieved);
      }

      // copy construction from a shared_future
      explicit
      _Future_impl(const shared_future<_Result>&);

      // move construction from a unique_future
      explicit
      _Future_impl(unique_future<_Result>&&);

      _State_ptr _M_state;
    };

  /// primary template for unique_future
  template<typename _Result>
    class unique_future : public _Future_impl<_Result>
    {
      typedef _Move_future_result<_Result> _Mover;

    public:
      /// Move constructor
      unique_future(unique_future&& __uf) : _Base_type(std::move(__uf)) { }

      // disable copying
      unique_future(const unique_future&) = delete;
      unique_future& operator=(const unique_future&) = delete;

      // retrieving the value
      typename _Mover::__rval_type
      get()
      { return _Mover::_S_move(this->_M_get_result()._M_value()); }

    private:
      typedef _Future_impl<_Result> _Base_type;
      typedef typename _Base_type::_State_ptr _State_ptr;

      friend class promise<_Result>;

      explicit
      unique_future(const _State_ptr& __state) : _Base_type(__state) { }
    };
 
  // partial specialization for unique_future<R&>
  template<typename _Result>
    class unique_future<_Result&> : public _Future_impl<_Result&>
    {
    public:
      /// Move constructor
      unique_future(unique_future&& __uf) : _Base_type(std::move(__uf)) { }

      // disable copying
      unique_future(const unique_future&) = delete;
      unique_future& operator=(const unique_future&) = delete;

      // retrieving the value
      _Result& get() { return *this->_M_get_result()._M_value_ptr; }

    private:
      typedef _Future_impl<_Result&>           _Base_type;
      typedef typename _Base_type::_State_ptr _State_ptr;

      friend class promise<_Result&>;

      explicit
      unique_future(const _State_ptr& __state) : _Base_type(__state) { }
    };

  // specialization for unique_future<void>
  template<>
    class unique_future<void> : public _Future_impl<void>
    {
    public:
      /// Move constructor
      unique_future(unique_future&& __uf) : _Base_type(std::move(__uf)) { }

      // disable copying
      unique_future(const unique_future&) = delete;
      unique_future& operator=(const unique_future&) = delete;

      // retrieving the value
      void get() { this->_M_get_result(); }

    private:
      typedef _Future_impl<void> _Base_type;
      typedef _Base_type::_State_ptr _State_ptr;

      friend class promise<void>;

      explicit
      unique_future(const _State_ptr& __state) : _Base_type(__state) { }
    };

  /// primary template for shared_future
  template<typename _Result>
    class shared_future : public _Future_impl<_Result>
    {
    public:
      /// Copy constructor
      shared_future(const shared_future& __sf) : _Base_type(__sf) { }

      /// Construct from a unique_future rvalue
      shared_future(unique_future<_Result>&& __uf)
      : _Base_type(std::move(__uf))
      { }

      shared_future& operator=(const shared_future&) = delete;

      // retrieving the value
      const _Result&
      get()
      { return this->_M_get_result()._M_value(); }

    private:
      typedef _Future_impl<_Result> _Base_type;
    };
 
  // partial specialization for shared_future<R&>
  template<typename _Result>
    class shared_future<_Result&> : public _Future_impl<_Result&>
    {
    public:
      /// Copy constructor
      shared_future(const shared_future& __sf) : _Base_type(__sf) { }

      /// Construct from a unique_future rvalue
      shared_future(unique_future<_Result&>&& __uf)
      : _Base_type(std::move(__uf))
      { }

      shared_future& operator=(const shared_future&) = delete;

      // retrieving the value
      _Result& get() { return *this->_M_get_result()._M_value_ptr; }

    private:
      typedef _Future_impl<_Result&>           _Base_type;
    };

  // specialization for shared_future<void>
  template<>
    class shared_future<void> : public _Future_impl<void>
    {
    public:
      /// Copy constructor
      shared_future(const shared_future& __sf) : _Base_type(__sf) { }

      /// Construct from a unique_future rvalue
      shared_future(unique_future<void>&& __uf)
      : _Base_type(std::move(__uf))
      { }

      shared_future& operator=(const shared_future&) = delete;

      // retrieving the value
      void get() { this->_M_get_result(); }

    private:
      typedef _Future_impl<void> _Base_type;
    };

  // now we can define the protected _Future_impl constructors

  template<typename _Result>
    _Future_impl<_Result>::_Future_impl(const shared_future<_Result>& __sf)
    : _M_state(__sf._M_state)
    { }

  template<typename _Result>
    _Future_impl<_Result>::_Future_impl(unique_future<_Result>&& __uf)
    : _M_state(std::move(__uf._M_state))
    { }

  /// primary template for promise
  template<typename _Result>
    class promise
    {
    public:
      promise()
      : _M_future(std::make_shared<_Future_state>()),
      _M_storage(new _Future_result<_Result>())
      { }

      promise(promise&& __rhs)
      : _M_future(std::move(__rhs._M_future)),
      _M_storage(std::move(__rhs._M_storage))
      { }

      // TODO: requires allocator concepts
      /*
      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator& __a);

      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator&, promise&& __rhs);
       */

      promise(const promise&) = delete;

      ~promise()
      {
        if (static_cast<bool>(_M_future) && !_M_future.unique())
          _M_future->_M_break_promise(std::move(_M_storage));
      }

      // assignment
      promise&
      operator=(promise&& __rhs)
      {
        promise(std::move(__rhs)).swap(*this);
        return *this;
      }

      promise& operator=(const promise&) = delete;

      void
      swap(promise& __rhs)
      {
        _M_future.swap(__rhs._M_future);
        _M_storage.swap(__rhs._M_storage);
      }

      // retrieving the result
      unique_future<_Result>
      get_future()
      { return unique_future<_Result>(_M_future); }

      // setting the result
      void
      set_value(const _Result& __r)
      {
        if (!_M_satisfied())
          _M_storage->_M_set(__r);
        _M_future->_M_set_result(std::move(_M_storage));
      }

      void
      set_value(_Result&& __r)
      {
        if (!_M_satisfied())
          _M_storage->_M_set(_Mover::_S_move(__r));
        _M_future->_M_set_result(std::move(_M_storage));
      }

      void
      set_exception(exception_ptr __p)
      {
        if (!_M_satisfied())
          _M_storage->_M_error = __p;
        _M_future->_M_set_result(std::move(_M_storage));
      }

    private:
      template<typename> friend class packaged_task;
      typedef _Move_future_result<_Result> _Mover;
      bool _M_satisfied() { return !static_cast<bool>(_M_storage); }
      shared_ptr<_Future_state>                           _M_future;
      typename _Future_ptr<_Future_result<_Result>>::type _M_storage;
    };

  // partial specialization for promise<R&>
  template<typename _Result>
    class promise<_Result&>
    {
    public:
      promise()
      : _M_future(std::make_shared<_Future_state>()),
      _M_storage(new _Future_result<_Result&>())
      { }

      promise(promise&& __rhs)
      : _M_future(std::move(__rhs._M_future)),
      _M_storage(std::move(__rhs._M_storage))
      { }

      // TODO: requires allocator concepts
      /*
      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator& __a);

      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator&, promise&& __rhs);
       */

      promise(const promise&) = delete;

      ~promise()
      {
        if (static_cast<bool>(_M_future) && !_M_future.unique())
          _M_future->_M_break_promise(std::move(_M_storage));
      }

      // assignment
      promise&
      operator=(promise&& __rhs)
      {
        promise(std::move(__rhs)).swap(*this);
        return *this;
      }

      promise& operator=(const promise&) = delete;

      void
      swap(promise& __rhs)
      {
        _M_future.swap(__rhs._M_future);
        _M_storage.swap(__rhs._M_storage);
      }

      // retrieving the result
      unique_future<_Result&>
      get_future()
      { return unique_future<_Result&>(_M_future); }

      // setting the result
      void
      set_value(_Result& __r)
      {
        if (!_M_satisfied())
          _M_storage->_M_value_ptr = &__r;
        _M_future->_M_set_result(std::move(_M_storage));
      }

      void
      set_exception(exception_ptr __p)
      {
        if (!_M_satisfied())
          _M_storage->_M_error = __p;
        _M_future->_M_set_result(std::move(_M_storage));
      }

    private:
      template<typename> friend class packaged_task;
      bool _M_satisfied() { return !static_cast<bool>(_M_storage); }
      shared_ptr<_Future_state>                             _M_future;
      typename _Future_ptr<_Future_result<_Result&>>::type  _M_storage;
    };

  // specialization for promise<void>
  template<>
    class promise<void>
    {
    public:
      promise()
      : _M_future(std::make_shared<_Future_state>()),
      _M_storage(new _Future_result<void>())
      { }

      promise(promise&& __rhs)
      : _M_future(std::move(__rhs._M_future)),
      _M_storage(std::move(__rhs._M_storage))
      { }

      // TODO: requires allocator concepts
      /*
      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator& __a);

      template<typename _Allocator>
        promise(allocator_arg_t, const _Allocator&, promise&& __rhs);
       */

      promise(const promise&) = delete;

      ~promise()
      {
        if (static_cast<bool>(_M_future) && !_M_future.unique())
          _M_future->_M_break_promise(std::move(_M_storage));
      }

      // assignment
      promise&
      operator=(promise&& __rhs)
      {
        promise(std::move(__rhs)).swap(*this);
        return *this;
      }

      promise& operator=(const promise&) = delete;

      void
      swap(promise& __rhs)
      {
        _M_future.swap(__rhs._M_future);
        _M_storage.swap(__rhs._M_storage);
      }

      // retrieving the result
      unique_future<void>
      get_future()
      { return unique_future<void>(_M_future); }

      // setting the result
      void
      set_value()
      {
        _M_future->_M_set_result(std::move(_M_storage));
      }

      void
      set_exception(exception_ptr __p)
      {
        if (!_M_satisfied())
          _M_storage->_M_error = __p;
        _M_future->_M_set_result(std::move(_M_storage));
      }

    private:
      template<typename> friend class packaged_task;
      bool _M_satisfied() { return !static_cast<bool>(_M_storage); }
      shared_ptr<_Future_state>                 _M_future;
      _Future_ptr<_Future_result<void>>::type   _M_storage;
    };

  // TODO: requires allocator concepts
  /*
  template<typename _Result, class Alloc>
    concept_map UsesAllocator<promise<_Result>, Alloc>
    {
      typedef Alloc allocator_type;
    }
   */

  template<typename _Result, typename... _ArgTypes>
    struct _Run_task
    {
      static void
      _S_run(promise<_Result>& __p, function<_Result(_ArgTypes...)>& __f,
          _ArgTypes... __args)
      {
        __p.set_value(__f(std::forward<_ArgTypes>(__args)...));
      }
    };

  // specialization used by packaged_task<void(...)>
  template<typename... _ArgTypes>
    struct _Run_task<void, _ArgTypes...>
    {
      static void
      _S_run(promise<void>& __p, function<void(_ArgTypes...)>& __f,
          _ArgTypes... __args)
      {
        __f(std::forward<_ArgTypes>(__args)...);
        __p.set_value();
      }
    };

  /// packaged_task
  template<typename _Result, typename... _ArgTypes>
    class packaged_task<_Result(_ArgTypes...)>
    {
    public:
      typedef _Result result_type;

      // construction and destruction
      packaged_task() { }

      template<typename _Fn>
        explicit
        packaged_task(const _Fn& __fn) : _M_task(__fn) { }

      template<typename _Fn>
        explicit
        packaged_task(_Fn&& __fn) : _M_task(std::move(__fn)) { }

      explicit
      packaged_task(_Result(*__fn)(_ArgTypes...)) : _M_task(__fn) { }

      // TODO: requires allocator concepts
      /*
      template<typename _Fn, typename _Allocator>
        explicit
        packaged_task(allocator_arg_t __tag, const _Allocator& __a, _Fn __fn)
        : _M_task(__tag, __a, __fn), _M_promise(__tag, __a)
        { }

      template<typename _Fn, typename _Allocator>
        explicit
        packaged_task(allocator_arg_t __tag, const _Allocator& __a, _Fn&& __fn)
        : _M_task(__tag, __a, std::move(__fn)), _M_promise(__tag, __a)
        { }
       */

      ~packaged_task() = default;

      // no copy
      packaged_task(packaged_task&) = delete;
      packaged_task& operator=(packaged_task&) = delete;

      // move support
      packaged_task(packaged_task&& __other)
      { this->swap(__other); }

      packaged_task& operator=(packaged_task&& __other)
      {
        packaged_task(std::move(__other)).swap(*this);
        return *this;
      }

      void
      swap(packaged_task& __other)
      {
        _M_task.swap(__other._M_task);
        _M_promise.swap(__other._M_promise);
      }

      operator bool() const { return static_cast<bool>(_M_task); }

      // result retrieval
      unique_future<_Result>
      get_future()
      {
        __try
        {
          return _M_promise.get_future();
        }
        __catch (const future_error& __e)
        {
#ifdef __EXCEPTIONS
          if (__e.code() == future_errc::future_already_retrieved)
	    throw std::bad_function_call();
	  throw;
#endif
        }
      }

      // execution
      void
      operator()(_ArgTypes... __args)
      {
        if (!static_cast<bool>(_M_task) || _M_promise._M_satisfied())
	  {
#ifdef __EXCEPTIONS
	    throw std::bad_function_call();
#else
	    __builtin_abort();
#endif
	  }

        __try
        {
          _Run_task<_Result, _ArgTypes...>::_S_run(_M_promise, _M_task,
              std::forward<_ArgTypes>(__args)...);
        }
        __catch (...)
        {
          _M_promise.set_exception(current_exception());
        }
      }

      void reset() { promise<_Result>().swap(_M_promise); }

    private:
      function<_Result(_ArgTypes...)>   _M_task;
      promise<_Result>                  _M_promise;
    };

#endif // _GLIBCXX_HAS_GTHREADS && _GLIBCXX_USE_C99_STDINT_TR1
       // && _GLIBCXX_ATOMIC_BUILTINS_4

  // @} group futures
}

#endif // __GXX_EXPERIMENTAL_CXX0X__

#endif // _GLIBCXX_FUTURE
