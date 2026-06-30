# Agent Notes

## Remote Server

Use the SSH alias `school13` as the default remote server for WoTE work.

Expected SSH config entry:

```sshconfig
Host school13
    HostName 166.111.50.13
    User zhaodanqi
    IdentitiesOnly yes
```

Default remote WoTE repository path:

```text
/home/zhaodanqi/clone/WoTE
```

When a task modifies files that correspond to this remote WoTE repository,
connect to `school13` and sync/apply the changes to the remote path unless the
user explicitly says the work should remain local only.
