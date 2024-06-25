//
// Updated on 22.05.2024
//
#include <cstddef>
#include <sycl/sycl.hpp>
#include <random>

using real_t = float;

void power(size_t len, sycl::queue &queue);
void exponent(size_t len, sycl::queue &queue);

int main(int argc, char* argv[])
{
    sycl::queue queue({ sycl::property::queue::enable_profiling() });

    std::cout << "Device name: " << queue.get_device().get_info<sycl::info::device::name>() << "\n";
    bool has_profiling = queue.get_device().has(sycl::aspect::queue_profiling);
    if (!has_profiling)
    {
        std::cout << "Device does not support profiling with events!\n";
    }

    size_t len = 800000;
    if (argc > 1)
    {
        len = std::atol(argv[1]);
    }
    std::cout << "Total data volume: " << len * sizeof(real_t) * 2 << " bytes.\n";

    power(len, queue);
    exponent(len, queue);
}

void power(size_t len, sycl::queue& queue)
{
    sycl::buffer<real_t, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<real_t, 1> b{ sycl::range<1>{ len }};

    std::default_random_engine rand;
    std::uniform_real_distribution<real_t> dist(0.0f, 10.0f);

    {
        auto acc = a.get_host_access();
        for (size_t i = 0; i < len; i++)
        {
            acc[i] = dist(rand);
        }
    }

    queue.submit([&](auto& handler)
    {
        auto a_acc = a.get_access<sycl::access_mode::read>(handler);
        auto b_acc = b.get_access<sycl::access_mode::discard_write>(handler);
        handler.parallel_for(sycl::nd_range<1>{{ len }, { 80 }}, [=](sycl::nd_item<1> id)
        {
            if (id.get_global_id() < len)
            {
                b_acc[id.get_global_id()] = sycl::pow(a_acc[id.get_global_id()], static_cast<real_t>(2.0));
            }
        });
    }).wait();

    {
        auto original_values = a.get_host_access();
        auto sycl_results = b.get_host_access();
        real_t max_diff_sycl = 0.0;
        real_t max_diff_std = 0.0;
        for (size_t i = 0; i < len; i++)
        {
            max_diff_sycl = std::max(
                std::abs(sycl_results[i] - sycl::pow(original_values[i], static_cast<real_t>(2.0))),
                max_diff_sycl);
            max_diff_std = std::max(
                std::abs(sycl_results[i] - std::pow(original_values[i], static_cast<real_t>(2.0))),
                max_diff_std);
            //real_t r = std::pow(original_values[i], 2.0);
	        //long n1, n2;
	        //std::memcpy(&n1, &r, sizeof(real_t));
	        //std::memcpy(&n2, &sycl_results[i], sizeof(real_t));
	        //std::cout << n1 - n2 << std::endl;
        }
        std::cout << "-------------- POW FUNCTION ----------------------"<<std::endl;
        std::cout << "Maximum difference between SYCL and CPU(SYCL): " << max_diff_sycl
                  << "\n";
        std::cout << "Maximum difference between SYCL and CPU(STD): " << max_diff_std
                  << "\n";
    }
}

void exponent(size_t len, sycl::queue& queue)
{
    sycl::buffer<real_t, 1> a{ sycl::range<1>{ len }};
    sycl::buffer<real_t, 1> b{ sycl::range<1>{ len }};

    std::default_random_engine rand;
    std::uniform_real_distribution<real_t> dist(0.0f, 10.0f);

    {
        auto acc = a.get_host_access();
        for (size_t i = 0; i < len; i++)
        {
            acc[i] = dist(rand);
        }
    }

    queue.submit([&](auto& handler)
    {
        auto a_acc = a.get_access<sycl::access_mode::read>(handler);
        auto b_acc = b.get_access<sycl::access_mode::discard_write>(handler);
        handler.parallel_for(sycl::nd_range<1>{{ len }, { 80 }}, [=](sycl::nd_item<1> id)
        {
            if (id.get_global_id() < len)
            {
                b_acc[id.get_global_id()] = sycl::exp(a_acc[id.get_global_id()]);
            }
        });
    }).wait();
 
    {
        auto original_values = a.get_host_access();
        auto sycl_results = b.get_host_access();
        real_t max_diff_sycl = 0.0;
        real_t max_diff_std = 0.0;
        for (size_t i = 0; i < len; i++)
        {
            max_diff_sycl = std::max(
                std::abs(sycl_results[i] - sycl::exp(original_values[i])),
                max_diff_sycl);
            max_diff_std = std::max(
                std::abs(sycl_results[i] - std::exp(original_values[i])),
                max_diff_std);
            //real_t r = std::pow(original_values[i], 2.0);
	        //long n1, n2;
	        //std::memcpy(&n1, &r, sizeof(real_t));
	        //std::memcpy(&n2, &sycl_results[i], sizeof(real_t));
	        //std::cout << n1 - n2 << std::endl;
        }
        std::cout << "-------------- EXP FUNCTION ----------------------"<<std::endl;
        std::cout << "Maximum difference between SYCL and CPU(SYCL): " << max_diff_sycl
                  << "\n";
        std::cout << "Maximum difference between SYCL and CPU(STD): " << max_diff_std
                  << "\n";
    }
}

