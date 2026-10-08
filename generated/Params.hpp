//  To parse this JSON data, first install
//
//      Boost     http://www.boost.org
//      json.hpp  https://github.com/nlohmann/json
//
//  Then include this file, and then do
//
//     Params.hpp data = nlohmann::json::parse(jsonString);

#pragma once

#include <boost/optional.hpp>
#include <nlohmann/json.hpp>
#include "helper.hpp"

namespace sgns {
    /**
     * Transform-specific parameters
     */

    using nlohmann::json;

    /**
     * Transform-specific parameters
     */
    class Params {
        public:
        Params() = default;
        virtual ~Params() = default;

        private:
        boost::optional<double> angle;
        boost::optional<std::vector<int64_t>> axes;
        boost::optional<std::string> color_space;
        boost::optional<std::string> custom_function;
        boost::optional<int64_t> height;
        boost::optional<std::vector<double>> mean;
        boost::optional<std::string> method;
        boost::optional<std::vector<double>> std;
        boost::optional<int64_t> width;

        public:
        const boost::optional<double> & get_angle() const { return angle; }
        boost::optional<double> & get_mutable_angle() { return angle; }
        void set_angle(const boost::optional<double> & value) { this->angle = value; }

        const boost::optional<std::vector<int64_t>> & get_axes() const { return axes; }
        boost::optional<std::vector<int64_t>> & get_mutable_axes() { return axes; }
        void set_axes(const boost::optional<std::vector<int64_t>> & value) { this->axes = value; }

        const boost::optional<std::string> & get_color_space() const { return color_space; }
        boost::optional<std::string> & get_mutable_color_space() { return color_space; }
        void set_color_space(const boost::optional<std::string> & value) { this->color_space = value; }

        const boost::optional<std::string> & get_custom_function() const { return custom_function; }
        boost::optional<std::string> & get_mutable_custom_function() { return custom_function; }
        void set_custom_function(const boost::optional<std::string> & value) { this->custom_function = value; }

        const boost::optional<int64_t> & get_height() const { return height; }
        boost::optional<int64_t> & get_mutable_height() { return height; }
        void set_height(const boost::optional<int64_t> & value) { this->height = value; }

        const boost::optional<std::vector<double>> & get_mean() const { return mean; }
        boost::optional<std::vector<double>> & get_mutable_mean() { return mean; }
        void set_mean(const boost::optional<std::vector<double>> & value) { this->mean = value; }

        const boost::optional<std::string> & get_method() const { return method; }
        boost::optional<std::string> & get_mutable_method() { return method; }
        void set_method(const boost::optional<std::string> & value) { this->method = value; }

        const boost::optional<std::vector<double>> & get_std() const { return std; }
        boost::optional<std::vector<double>> & get_mutable_std() { return std; }
        void set_std(const boost::optional<std::vector<double>> & value) { this->std = value; }

        const boost::optional<int64_t> & get_width() const { return width; }
        boost::optional<int64_t> & get_mutable_width() { return width; }
        void set_width(const boost::optional<int64_t> & value) { this->width = value; }
    };
}
