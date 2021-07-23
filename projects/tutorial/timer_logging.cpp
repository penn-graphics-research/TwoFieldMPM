#include <Bow/Utils/Logging.h>
#include <Bow/Utils/Timer.h>

int main()
{
    // a new logger only logs message with or above info level
    Bow::Logging::new_logger("test.txt", Bow::Logging::Info);
    // a new logger only logs message with or above error level
    Bow::Logging::new_logger("test2.txt", Bow::Logging::Error);
    // log a string
    Bow::Logging::info("test!");
    // you can log multiple objects as long as they support << operator;
    Bow::Logging::info(1, "+", 1, "=", 2);
    // you won't see this message in your command line
    Bow::Logging::debug("debug test!");
    // change the log level of stdout logger
    Bow::Logging::set_level(Bow::Logging::Debug);
    // now you can see this message
    Bow::Logging::debug("debug test again!");
    Bow::Logging::warn("warning!");
    Bow::Logging::timing("timing!");
    Bow::Logging::error("error!");

    // herachical timer example
    {
        BOW_TIMER_FLAG("level 0");
        for (int i = 0; i < 10; ++i) {
            Bow::Timer::progress("Level 0 - " + std::to_string(i), i, 10);
        }
        {
            BOW_TIMER_FLAG("level 1");
            for (int i = 0; i < 10; ++i) {
                Bow::Timer::progress("Level 1 - " + std::to_string(i), i, 10);
            }
            {
                BOW_TIMER_FLAG("level 2");
                for (int i = 0; i < 10; ++i) {
                    Bow::Timer::progress("Level 2 - " + std::to_string(i), i, 10);
                }
            }
        }
    }

    Bow::Timer::flush();

    // raise a runtime_error
    // Bow::Logging::fatal(std::string("fatal!"));
}