# Security

## Report a security issue

Meta has a [bounty program](https://bugbounty.meta.com/) for the safe
disclosure of security bugs. If you find a security issue, use the process on
that page. Do not open a public issue.

## Trust model

torchpyimpl is for ranks that trust each other, for example the processes of
one training or inference job. It does not protect a rank from a malicious
peer.

- All ranks in a process group must be trusted. torchpyimpl exchanges data
  with `torch.distributed.all_gather_object`, which uses `pickle`. A malicious
  rank can send data that runs code in a peer when the peer unpickles it.
- Symmetric memory gives each rank read and write access to the buffers of its
  peers. Thus, one rank can change the data of all its peers.
- All ranks on a node must run as the same user. The `fd` transport sends file
  descriptors through Unix sockets. Each socket is in a private directory
  (mode 0700), and other users cannot get access to it.
- The rendezvous store (for example `TCPStore`) and the network between nodes
  must be trusted. Use the controls of `torch.distributed` and of your cluster
  to protect them.
