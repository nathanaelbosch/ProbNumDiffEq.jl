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


function IIP_to_OOP(f_iip)
    function f_oop(u, p, t)
        println("Hello")
        println(size(u))
        du = _copy(u)
        f_iip(du, u, p, t)
        return du
    end
    return f_oop
end
