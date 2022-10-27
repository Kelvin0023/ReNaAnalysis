plt.stem(block_id_timestamps, block_ids)
markerlines, _, _ = plt.stem(dtn_timestamps, dtn)
markerlines.set_markerfacecolor('orange')
plt.show()