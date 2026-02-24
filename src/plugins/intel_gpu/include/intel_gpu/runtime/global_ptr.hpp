#include <iostream>
#include <memory>

namespace cldnn {
    inline cldnn::memory::ptr& get_tracked_ptr() {
        static cldnn::memory::ptr instance = nullptr;
        return instance;
    }
    
    inline void set_tracked_ptr(cldnn::memory::ptr ptr) {
        if (get_tracked_ptr() == nullptr) {
            get_tracked_ptr() = ptr;
        } 
    }

    [[maybe_unused]] static void print_tracked_ptr(const cldnn::stream& stream, std::string name,int size = 10) {
        auto g_tracked_ptr = get_tracked_ptr();
        if (g_tracked_ptr == nullptr) {
            std::cout << "Tracked pointer is null." << std::endl;
            return;
        }
        std::cout << "Tracked pointer name: " << name << std::endl;
        std::cout << "Tracked pointer address: " << g_tracked_ptr.get() << std::endl;
        cldnn::mem_lock<ov::float16, mem_lock_type::read> print_ptr(g_tracked_ptr, stream);
        std::cout << "wzx debug scale[0] gpu data2:" ;
        for (int i = 0; i < size; i++) {
            std::cout << "[" << i << "]: " << print_ptr[i] << " ";
        }   
        std::cout << std::endl;
    }
}

