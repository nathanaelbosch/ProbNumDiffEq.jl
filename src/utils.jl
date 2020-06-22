using UUIDs
using ProgressLogging
# using Logging: global_logger
using TerminalLoggers: TerminalLogger


function make_progressbar(progress_every)
    id = uuid4()
    @info ProgressLogging.Progress(id)
    next_threshold = progress_every
    function update(; fraction)
        if fraction >= next_threshold
            @info ProgressLogging.Progress(id, fraction)
            next_threshold += progress_every
        end
    end
    close() = @info ProgressLogging.Progress(id, done=true)
    return update, close
end
