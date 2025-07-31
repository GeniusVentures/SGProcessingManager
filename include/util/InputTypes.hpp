#ifndef INPUTTYPES_HPP_
#define INPUTTYPES_HPP_
#include <InputFormat.hpp>
#include <outcome/sgprocmgr-outcome.hpp>

namespace sgns::sgprocessing
{
    class InputTypes
    {
    public:
        enum class Error
        {
            NOT_IMAGE_TYPE = 1,
            NOT_FLOAT_TYPE = 2,
        };

        InputTypes();

        ~InputTypes();

        static outcome::result<int> GetImageChannels( sgns::InputFormat format );

    private:

    };
}


#endif