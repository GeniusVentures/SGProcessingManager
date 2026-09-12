// ELM stop-string streambuf unit tests (elmbridge Phase 3, plan 03-03, Task 1).
//
// D-05/D-06/D-07 semantics at unit level (no MNN -- the cancel hook is an
// injectable std::function): exclusion of the stop string from VisibleText,
// split-across-writes matching, earliest-match-wins, no-match pass-through,
// the overlap-window negative leg, and the D-09 two-latch disambiguation
// (stop-string latch vs external-cancel latch fire the same one-shot hook).

#include <gtest/gtest.h>

#include <elmruntime/ElmStopStringStreamBuf.hpp>

#include <atomic>
#include <string>
#include <vector>

namespace
{
    using sgns::elmruntime::ElmStopStringStreamBuf;

    struct CancelCount
    {
        std::atomic<int> fires{ 0 };
        void Hook()
        {
            ++fires;
        }
    };

    ElmStopStringStreamBuf MakeBuf( const std::vector<std::string> &stops, CancelCount &counter )
    {
        return ElmStopStringStreamBuf( stops, [ &counter ]() { counter.Hook(); } );
    }
} // namespace

// Basic match + D-07 exclusion: "Hello WORLD tail" with stop "WORLD" matches
// at offset 6 and VisibleText() == "Hello " (stop string NOT included).
TEST( ElmStopStreamBufTest, MatchExcludesStopString )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "WORLD" }, counter );

    buf.sputn( "Hello WORLD tail", 16 );

    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 6 );
    EXPECT_EQ( buf.VisibleText(), "Hello " );
    EXPECT_EQ( buf.AccumulatedText(), "Hello WORLD tail" );
    EXPECT_EQ( counter.fires.load(), 1 );
}

// Split across two writes (the token-boundary case D-06 exists for): the
// stop string still matches.
TEST( ElmStopStreamBufTest, StopStringSplitAcrossWritesMatches )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "WORLD" }, counter );

    buf.sputn( "Hello WO", 8 );
    EXPECT_FALSE( buf.Matched() );

    buf.sputn( "RLD tail", 8 );
    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 6 );
    EXPECT_EQ( buf.VisibleText(), "Hello " );
}

// Two stop strings, only the later-listed one present: earliest match
// position wins (here only one candidate matches at all).
TEST( ElmStopStreamBufTest, EarliestMatchPositionWins )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "zebra", "WORLD" }, counter );

    buf.sputn( "Hello WORLD", 11 );

    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 6 );
}

// Two stop strings BOTH present: the earlier match position wins regardless
// of list order.
TEST( ElmStopStreamBufTest, EarliestOfTwoPresentMatches )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "tail", "WORLD" }, counter );

    buf.sputn( "Hello WORLD tail", 16 );

    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 6 );
    EXPECT_EQ( buf.VisibleText(), "Hello " );
}

// No match: Matched() false, VisibleText() == the full accumulation, hook
// never fired.
TEST( ElmStopStreamBufTest, NoMatchPassesEverythingThrough )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "zebra" }, counter );

    buf.sputn( "Hello WORLD tail", 16 );

    EXPECT_FALSE( buf.Matched() );
    EXPECT_EQ( buf.VisibleText(), "Hello WORLD tail" );
    EXPECT_EQ( counter.fires.load(), 0 );
}

// Overlap window negative leg: a stop string whose first half arrived long
// ago (beyond the window) and was never completed does not match on later
// text. A partial "WO" prefix lands, then enough filler scrolls it out of
// the window (a 4-char stop's window is 4-1+16 = 19 bytes; 40 filler bytes
// push "WO" out), then "RLD" arrives -- no match, because "WO" was never a
// complete stop string and the stale prefix is outside the scan window.
TEST( ElmStopStreamBufTest, StalePartialPrefixBeyondWindowDoesNotMatch )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "WORLD" }, counter );

    buf.sputn( "prefix WO", 9 );
    EXPECT_FALSE( buf.Matched() );

    buf.sputn( std::string( 40, '.' ).c_str(), 40 );
    buf.sputn( "RLD suffix", 9 );

    EXPECT_FALSE( buf.Matched() );
    EXPECT_EQ( counter.fires.load(), 0 );
}

// A partial prefix still INSIDE the window completes on a later write
// (the positive counterpart of the window leg).
TEST( ElmStopStreamBufTest, PartialPrefixInsideWindowCompletes )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "WORLD" }, counter );

    buf.sputn( "Hello WO", 8 );
    buf.sputn( "RL", 2 );
    EXPECT_FALSE( buf.Matched() );

    buf.sputn( "D", 1 );
    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 6 );
}

// Empty stop list: no scanning, everything accumulates, hook never fires.
TEST( ElmStopStreamBufTest, EmptyStopListAccumulatesOnly )
{
    CancelCount counter;
    auto        buf = MakeBuf( {}, counter );

    buf.sputn( "anything at all", 15 );

    EXPECT_FALSE( buf.Matched() );
    EXPECT_EQ( buf.VisibleText(), "anything at all" );
    EXPECT_EQ( counter.fires.load(), 0 );
}

// External cancel poll (D-09 seam): a true poll latches CANCEL intent (not
// the stop-string latch) and fires the SAME one-shot hook.
TEST( ElmStopStreamBufTest, ExternalCancelPollLatchesCancelIntent )
{
    CancelCount    counter;
    std::atomic<bool> cancelled{ false };
    auto           buf = MakeBuf( { "zebra" }, counter );
    buf.SetExternalCancelPoll( [ &cancelled ]() { return cancelled.load(); } );

    buf.sputn( "Hello", 5 );
    EXPECT_FALSE( buf.CancelRequested() );
    EXPECT_EQ( counter.fires.load(), 0 );

    cancelled.store( true );
    buf.sputn( " WORLD", 6 );

    EXPECT_TRUE( buf.CancelRequested() );
    EXPECT_FALSE( buf.Matched() ); // cancel intent, NOT a stop-string match
    EXPECT_EQ( counter.fires.load(), 1 );
}

// Idempotence: after either latch fires, subsequent writes never re-fire
// the hook (the cancel call is one-shot by design).
TEST( ElmStopStreamBufTest, CancelHookFiresOnlyOnce )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "WORLD" }, counter );

    buf.sputn( "Hello WORLD", 11 );
    ASSERT_TRUE( buf.Matched() );
    EXPECT_EQ( counter.fires.load(), 1 );

    buf.sputn( " more text", 10 );
    EXPECT_EQ( counter.fires.load(), 1 ); // still once
    EXPECT_EQ( buf.MatchOffset(), 6 );    // first match position is stable
}

// overflow() path: single-character puts behave like xsputn.
TEST( ElmStopStreamBufTest, OverflowPathMatches )
{
    CancelCount counter;
    auto        buf = MakeBuf( { "END" }, counter );

    for ( const char c : std::string( "text END" ) )
    {
        buf.sputc( c );
    }

    EXPECT_TRUE( buf.Matched() );
    EXPECT_EQ( buf.MatchOffset(), 5 );
    EXPECT_EQ( buf.VisibleText(), "text " );
}
