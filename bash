while :; do
   journalctl -u NetworkManager --since "10 seconds ago" --no-pager | grep -iE "p2p|wifi-p2p|cancel"
done