#include <util/InputTypes.hpp>

OUTCOME_CPP_DEFINE_CATEGORY_3( sgns::sgprocessing, InputTypes::Error, e )
{
    switch ( e )
    {
        case sgns::sgprocessing::InputTypes::Error::NOT_IMAGE_TYPE:
            return "Format was not an image format";
        case sgns::sgprocessing::InputTypes::Error::NOT_FLOAT_TYPE:
            return "Format was not a float format";
    }
    return "Unknown error";
}

namespace sgns::sgprocessing
{
    outcome::result<int> InputTypes::GetImageChannels( sgns::InputFormat format )
    {
        int channels;
        if ( format == sgns::InputFormat::RGB8 )
        {
            return 3;
        }
        else if ( format == sgns::InputFormat::RGBA8 )
        {
            return 4;
        }
        return outcome::failure( Error::NOT_IMAGE_TYPE );
    }
}
