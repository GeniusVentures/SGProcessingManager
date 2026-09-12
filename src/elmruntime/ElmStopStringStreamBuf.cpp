#include <elmruntime/ElmStopStringStreamBuf.hpp>

namespace sgns::elmruntime
{
    namespace
    {
        /// D-06 slack margin: how many bytes of history the incremental scan
        /// window keeps beyond a stop string's own length, so a stop string
        /// split across token boundaries (each token's decoded text arrives as
        /// a separate xsputn) still completes inside the window. 16 UTF-8 bytes
        /// (discretion-sanctioned; handles multi-token splits of typical stop
        /// strings).
        constexpr std::size_t kOverlapSlack = 16;
    } // namespace

    ElmStopStringStreamBuf::ElmStopStringStreamBuf( std::vector<std::string> stopStrings,
                                                    std::function<void()>    onCancelMatch )
        : stopStrings_( std::move( stopStrings ) )
        , onCancelMatch_( std::move( onCancelMatch ) )
    {
    }

    bool ElmStopStringStreamBuf::Matched() const
    {
        return stopStringMatched_.load( std::memory_order_acquire );
    }

    std::size_t ElmStopStringStreamBuf::MatchOffset() const
    {
        return matchOffset_;
    }

    const std::string &ElmStopStringStreamBuf::AccumulatedText() const
    {
        return text_;
    }

    std::string_view ElmStopStringStreamBuf::VisibleText() const
    {
        if ( stopStringMatched_.load( std::memory_order_acquire ) )
        {
            return std::string_view( text_ ).substr( 0, matchOffset_ );
        }
        return text_;
    }

    bool ElmStopStringStreamBuf::CancelRequested() const
    {
        return cancelRequested_.load( std::memory_order_acquire );
    }

    void ElmStopStringStreamBuf::SetExternalCancelPoll( std::function<bool()> poll )
    {
        externalCancelPoll_ = std::move( poll );
    }

    std::streamsize ElmStopStringStreamBuf::xsputn( const char *s, std::streamsize count )
    {
        AppendAndScan( s, count );
        return count;
    }

    ElmStopStringStreamBuf::int_type ElmStopStringStreamBuf::overflow( int_type ch )
    {
        if ( !traits_type::eq_int_type( ch, traits_type::eof() ) )
        {
            const char c = traits_type::to_char_type( ch );
            AppendAndScan( &c, 1 );
        }
        return traits_type::not_eof( ch );
    }

    void ElmStopStringStreamBuf::AppendAndScan( const char *s, std::streamsize count )
    {
        if ( count <= 0 )
        {
            return;
        }

        text_.append( s, static_cast<std::size_t>( count ) );

        // External cancel poll (Pitfall 4 resolution): checked at every append
        // (i.e., every token flush). A true return latches CANCEL intent --
        // distinct from the stop-string latch (D-09) -- and fires the same
        // one-shot cancel hook.
        if ( externalCancelPoll_ && !cancelRequested_.load( std::memory_order_acquire )
             && externalCancelPoll_() )
        {
            cancelRequested_.store( true, std::memory_order_release );
            FireCancelOnce();
        }

        if ( stopStrings_.empty() || stopStringMatched_.load( std::memory_order_acquire ) )
        {
            return; // no stop list, or already matched -- nothing to scan
        }

        // D-06 incremental overlap scan: a stop string of length L can only
        // START within the last (L - 1 + slack) bytes of the accumulated text.
        // EARLIEST MATCH POSITION wins across ALL stop strings (not list
        // order); the window start is byte-based and backed off to a UTF-8
        // lead byte so a sequence is never split.
        std::size_t bestPos    = std::string::npos;
        std::size_t bestWindow = 0;
        for ( const auto &stop : stopStrings_ )
        {
            if ( stop.empty() || stop.size() > text_.size() )
            {
                continue;
            }
            const std::size_t windowSize = stop.size() - 1 + kOverlapSlack;
            std::size_t       start      = text_.size() > windowSize ? text_.size() - windowSize : 0;
            start                         = BackOffToUtf8LeadByte( text_, start );

            const std::size_t pos = text_.find( stop, start );
            if ( pos != std::string::npos && ( bestPos == std::string::npos || pos < bestPos ) )
            {
                bestPos    = pos;
                bestWindow = windowSize;
            }
        }
        if ( bestPos != std::string::npos )
        {
            (void) bestWindow;
            stopStringMatched_.store( true, std::memory_order_release );
            matchOffset_ = bestPos;
            FireCancelOnce();
        }
    }

    void ElmStopStringStreamBuf::FireCancelOnce()
    {
        bool expected = false;
        if ( cancelFired_.compare_exchange_strong( expected, true, std::memory_order_acq_rel ) )
        {
            if ( onCancelMatch_ )
            {
                onCancelMatch_();
            }
        }
    }

    std::size_t ElmStopStringStreamBuf::BackOffToUtf8LeadByte( const std::string &text,
                                                               std::size_t        offset )
    {
        // A UTF-8 continuation byte is 10xxxxxx (0x80..0xBF); back the window
        // start off until it no longer sits on one (or reaches 0).
        while ( offset > 0 && ( static_cast<unsigned char>( text[ offset ] ) & 0xC0 ) == 0x80 )
        {
            --offset;
        }
        return offset;
    }
} // namespace sgns::elmruntime
