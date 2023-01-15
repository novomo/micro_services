const tiktok = async () => {};

const youtube = async () => {};

const twitter = async () => {};

const twitterVideo = async () => {};

const twitterLive = async () => {};

const instagram = async () => {};

const facebook = async () => {};

const facebookWatch = async () => {};

const steemit = async () => {};

const lbry = async () => {};

const dTube = async () => {};

const dLive = async () => {};

const minds = async () => {};

const newVideo = async () => {};

const tip = async () => {};

const liveStream = async () => {};

const testPost = async () => {};

const imagePost = async () => {};

exports.handler = async (event, context, callback) => {
  const jsonData = JSON.parse(event.body);

  callback(null, {
    statsusCode: 200,
    body: JSON.stringify({ return: true }),
  });
};
