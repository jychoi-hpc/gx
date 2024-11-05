
#include <numeric>
#include <vector>

// Template classes for implement Pollard's Rho algorithm
// for prime factorization. Required to check if FFTs will fail / be very slow

// This function finds a factor of n, or fails (one then has to retry with a new 'start' value)
// Starting with 2 is often sufficient
template<typename I> I pollard( I n, I start )
{
    I x = start;
    I y = x;
    I d = 1;
    auto g = [n](I z){ return ( z*z + 1 ) % n ; };
    do {
        x = g(x);
        y = g(g(y));
        d = std::gcd( abs(x-y), n );
    } while ( d == 1 );

    if( d == n )
        return 0;
    else 
        return d;
}

// This function takes a target n and finds a factor (which it returns) or
// declares the input to be a prime number for our purposes.
//
// Implemented as a wrapper of pollard<I> above including the retries.
// We assume after a certain amount of retries that n is in fact prime
template<typename I> I pollard_retry( I n )
{
    for( I x = 2; x < 10; ++x ) {
        I d = pollard( n, x );
        if( d > 0 )
            return d;
    }
    return 0;
}

// Returns a std::vector<I> containing the prime factors of target.
//
// Implemented by using pollard_retry<I> to get a factor, dividing that out and 
// then using pollard_retry<I> again until we get to a prime.
template<typename I> std::vector<I> factorize( I target )
{
    std::vector<I> factors{};
    if( target == 1 )
        return {1};
    I n = target;
    do {
        I d = pollard_retry( n );
        if( d == 0 ) {
            // n is at this point prime, so we have all factors
            factors.emplace_back( n );
            return factors;
        }
        if( d == 1 )
            throw std::logic_error("Impossible return from pollard_retry inside factorize");
        factors.emplace_back( d );
        n /= d;
    } while( true );
}
