import tensorflow as tf

# Define the path to your .record file
record_file = 'day_time_wildfire_v2/train.record'

# Create a TFRecordDataset to read the file
dataset = tf.data.TFRecordDataset(record_file)

# Iterate through the dataset and print each record
cnt = 0
for record in dataset:
    if cnt == 1:
        break
    # Parse the record (if it contains serialized data)
    # For example, if your records contain serialized TFExample protos:
    # parsed_record = tf.train.Example.FromString(record.numpy())
    
    # Process or print the parsed record
    # Convert hexadecimal string to bytes
    bytes_data = bytes.fromhex(record.numpy().hex())
    print(decoded_data)
    cnt += 1
 