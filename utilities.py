# Borrowed from https://github.com/PyTorchLightning/pytorch-lightning/blob/master/pytorch_lightning/utilities/distributed.py#L35
def rank_zero_only(fn):
    @wraps(fn)
    def wrapped_fn(*args, **kwargs):
        if rank_zero_only.rank == 0:
            return fn(*args, **kwargs)

    return


# add the attribute to the function but don't overwrite in case Trainer has already set it
rank_zero_only.rank = getattr(
    rank_zero_only, "rank", int(os.environ.get("LOCAL_RANK", 0))
)
